from __future__ import annotations

from dataclasses import asdict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bouldering2d.config import EnvConfig, MuscleConfig, PlayerConfig, RewardConfig, StaminaConfig
from bouldering2d.environment.contacts import ContactManager
from bouldering2d.environment.holds import Hold, HoldField
from bouldering2d.environment.muscle import MuscleModel
from bouldering2d.environment.physics import PhysicsEngine
from bouldering2d.environment.player_model import JOINT_ORDER, JOINT_LIMITS, LimbEndpoints, PlayerModel
from bouldering2d.environment.renderer import RenderContext, Renderer
from bouldering2d.environment.reward import RewardModel
from bouldering2d.environment.stamina import StaminaModel


class BoulderingEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    nearest_hold_count = 100
    hold_detect_radius = 2.0
    visible_holds_per_limb = 5

    def __init__(self, render_mode: str | None = None, config: EnvConfig | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.config = config or EnvConfig()
        self.player_config = PlayerConfig()
        self.stamina_cfg = StaminaConfig()
        self.reward_cfg = RewardConfig()
        self.np_random = np.random.default_rng(self.config.seed)

        self.muscle_cfg = MuscleConfig()
        self.player = PlayerModel(self.player_config)
        self.muscle_model = MuscleModel(self.muscle_cfg)
        self.holds = HoldField(self.config, self.np_random)
        self.contacts = ContactManager(self.player_config)
        self.stamina_model = StaminaModel(self.config, self.stamina_cfg)
        self.reward_model = RewardModel(self.reward_cfg)
        self.physics = PhysicsEngine(self.config)
        self.renderer = Renderer(self.config)

        self.joint_action_dim = len(JOINT_ORDER)
        self.observation_dim = (
            6
            + (2 * len(JOINT_ORDER))
            + len(self.contacts.as_array())
            + 8                          # 4 contacts × (dx, dy) relative to pelvis
            + (2 * self.nearest_hold_count)
            + len(JOINT_ORDER)  # per-muscle fatigue
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.joint_action_dim + 4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(self.observation_dim,), dtype=np.float32)

        self.stamina = self.config.start_stamina
        self.initial_y = float(self.player.pelvis[1])
        self.step_count = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self.holds = HoldField(self.config, self.np_random)

        self.player.reset(x=self.config.wall_width * 0.5, y=0.8)
        self.muscle_model.reset()
        self.contacts.reset()
        self.holds.reset(float(self.player.pelvis[1]))
        self.contacts.update(np.ones(4, dtype=np.float32), self.player.limb_endpoints(), self.holds)
        self.stamina = self.config.start_stamina
        self.initial_y = float(self.player.pelvis[1])
        self.step_count = 0

        obs = self._build_obs()
        info = {"config": asdict(self.config)}
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        self.step_count += 1

        torque_actions = action[: self.joint_action_dim]
        grip_actions = action[self.joint_action_dim :]

        joint_result = self.player.apply_joint_actions(torque_actions, self.config.dt, self.muscle_model)
        self.muscle_model.step(joint_result.joint_torques, self.config.dt)
        effort = joint_result.total_effort
        endpoints = self.player.limb_endpoints()
        new_contacts = self.contacts.update(grip_actions, endpoints, self.holds)
        dy, fell = self.physics.step(self.player, self.contacts, endpoints)
        self.holds.visible_holds(float(self.player.pelvis[1]))
        stamina_result = self.stamina_model.update(
            stamina=self.stamina,
            effort=effort,
            attachments=self.contacts.state.attached_count(),
            dt=self.config.dt,
        )
        self.stamina = stamina_result.stamina

        terminated = bool(fell or self.stamina <= self.config.min_stamina)
        truncated = self.step_count >= self.config.max_steps

        ascent = float(self.player.pelvis[1] - self.initial_y)
        if ascent >= self.config.ascent_target:
            terminated = True

        stale_contacts = self.contacts.stale_count(float(self.player.pelvis[1]))
        reward_breakdown = self.reward_model.compute(
            dy=dy,
            stamina_spent=stamina_result.spent,
            new_contacts=new_contacts,
            attachments=self.contacts.state.attached_count(),
            fell=fell,
            stale_contacts=stale_contacts,
        )

        obs = self._build_obs()
        info = {
            "ascent": ascent,
            "stamina": self.stamina,
            "reward_components": {
                "ascent": reward_breakdown.ascent,
                "energy": reward_breakdown.energy,
                "contact": reward_breakdown.contact,
                "idle": reward_breakdown.idle,
                "fall": reward_breakdown.fall,
            },
            "fell": fell,
        }
        return obs, float(reward_breakdown.total), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        endpoints = self.player.limb_endpoints()
        ctx = RenderContext(
            player=self.player,
            endpoints=endpoints,
            holds=self.holds.visible_holds(float(self.player.pelvis[1])),
            observed_holds=self.holds.nearest_holds(
                float(self.player.pelvis[0]),
                float(self.player.pelvis[1]),
                k=self.nearest_hold_count,
                max_dist=self.hold_detect_radius,
            ),
            contacts=self.contacts,
            stamina=self.stamina,
            step_count=self.step_count,
            ascent=float(self.player.pelvis[1] - self.initial_y),
            muscle_fatigue=self.muscle_model.state,
        )
        return self.renderer.render(ctx, self.render_mode)

    def close(self):
        self.renderer.close()

    def _build_obs(self) -> np.ndarray:
        endpoints = self.player.limb_endpoints()
        holds = self.holds.nearest_holds(
            float(self.player.pelvis[0]),
            float(self.player.pelvis[1]),
            k=self.nearest_hold_count,
            max_dist=self.hold_detect_radius,
        )
        obs = []

        obs.extend([
            float((self.player.pelvis[0] / self.config.wall_width) * 2.0 - 1.0),
            float(self.player.pelvis[1] * 0.1),
            float(self.player.pelvis_velocity[0] * 0.3),
            float(self.player.pelvis_velocity[1] * 0.3),
            float((self.stamina / self.config.start_stamina) * 2.0 - 1.0),
            float((self.player.pelvis[1] - self.initial_y) * 0.1),
        ])

        for name in JOINT_ORDER:
            lo, hi = JOINT_LIMITS[name]
            span = hi - lo
            normalized = ((self.player.joint_angles[name] - lo) / span) * 2.0 - 1.0
            obs.append(float(normalized))
        for name in JOINT_ORDER:
            obs.append(float(np.tanh(self.player.joint_velocities[name] * 0.5)))

        obs.extend(self.contacts.as_array().tolist())
        # Contact positions relative to pelvis (0,0 if not attached).
        # Lets the agent know if a held hold is now far below it.
        obs.extend(self.contacts.as_position_obs(self.player.pelvis).tolist())

        for hold in holds:
            obs.append(float((hold.x - self.player.pelvis[0]) * 0.6))
            obs.append(float((hold.y - self.player.pelvis[1]) * 0.4))
        # Pad holds to fixed size
        hold_obs_end = 6 + (2 * len(JOINT_ORDER)) + len(self.contacts.as_array()) + 8 + (2 * self.nearest_hold_count)
        while len(obs) < hold_obs_end:
            obs.append(0.0)

        # Per-muscle fatigue: map [0, 1] → [-1, 1]
        for name in JOINT_ORDER:
            obs.append(float(self.muscle_model.state.fatigue[name] * 2.0 - 1.0))

        _ = endpoints
        return np.asarray(obs[: self.observation_dim], dtype=np.float32)

    def _highlight_holds_from_endpoints(self, endpoints: LimbEndpoints) -> list[Hold]:
        points = [
            endpoints.left_hand,
            endpoints.right_hand,
            endpoints.left_foot,
            endpoints.right_foot,
        ]
        unique: dict[tuple[float, float], Hold] = {}
        for point in points:
            nearest = self.holds.nearest_holds(
                float(point[0]),
                float(point[1]),
                k=self.visible_holds_per_limb,
            )
            for hold in nearest:
                unique[(hold.x, hold.y)] = hold
        return list(unique.values())


def make_env(render_mode: str | None = None, seed: int | None = None) -> BoulderingEnv:
    env = BoulderingEnv(render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env

from __future__ import annotations

from dataclasses import asdict

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from bouldering2d.config import EnvConfig, MuscleConfig, PlayerConfig, RewardConfig
from bouldering2d.environment.contacts import ContactManager
from bouldering2d.environment.holds import Hold, HoldField
from bouldering2d.environment.muscle import MuscleModel
from bouldering2d.environment.physics import PhysicsEngine
from bouldering2d.environment.player_model import JOINT_ORDER, JOINT_LIMITS, LimbEndpoints, PlayerModel
from bouldering2d.environment.renderer import RenderContext, Renderer
from bouldering2d.environment.reward import RewardModel


class BoulderingEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    visible_holds_per_limb = 5

    def __init__(self, render_mode: str | None = None, config: EnvConfig | None = None):
        super().__init__()
        self.render_mode = render_mode
        self.config = config or EnvConfig()
        self.player_config = PlayerConfig()
        self.np_random = np.random.default_rng(self.config.seed)

        self.muscle_cfg = MuscleConfig()
        self.reward_cfg = RewardConfig()
        self.player = PlayerModel(self.player_config)
        self.muscle_model = MuscleModel(self.muscle_cfg)
        self.holds = HoldField(self.config, self.np_random)
        self.contacts = ContactManager(self.player_config)
        self.reward_model = RewardModel(self.reward_cfg)
        self.physics = PhysicsEngine(self.config)
        self.renderer = Renderer(self.config)
        self._prev_reach_potential: float = 0.0

        self.joint_action_dim = len(JOINT_ORDER)
        self.observation_dim = (
            5                                           # pelvis state
            + (2 * len(JOINT_ORDER))                   # joint angles + velocities
            + 4                                        # contact flags
            + 8                                        # contact positions relative to pelvis
            + (4 * self.visible_holds_per_limb * 3)   # per-limb holds: (dx, dy, in_reach)
            + len(JOINT_ORDER)                         # per-muscle fatigue
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.joint_action_dim + 4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(self.observation_dim,), dtype=np.float32)

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
        self.initial_y = float(self.player.pelvis[1])
        self.step_count = 0
        self._prev_reach_potential = self._reach_potential(self.player.limb_endpoints())

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
        endpoints = self.player.limb_endpoints()
        new_contacts = self.contacts.update(grip_actions, endpoints, self.holds)
        dy, fell = self.physics.step(self.player, self.contacts, endpoints)
        self.holds.visible_holds(float(self.player.pelvis[1]))

        # Potential-based reach shaping: reward free limbs moving toward holds
        curr_endpoints = self.player.limb_endpoints()
        curr_potential = self._reach_potential(curr_endpoints)
        reach_delta = self.reward_cfg.reach_shaping_weight * (curr_potential - self._prev_reach_potential)
        self._prev_reach_potential = curr_potential

        terminated = bool(fell)
        truncated = self.step_count >= self.config.max_steps

        ascent = float(self.player.pelvis[1] - self.initial_y)

        reward_breakdown = self.reward_model.compute(
            dy=dy,
            stamina_spent=joint_result.total_effort,
            new_contacts=new_contacts,
            attachments=self.contacts.state.attached_count(),
            fell=fell,
            stale_contacts=self.contacts.stale_count(float(self.player.pelvis[1]), self.reward_cfg.stale_threshold),
            reach_delta=reach_delta,
        )

        obs = self._build_obs()
        info = {
            "ascent": ascent,
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
            observed_holds=self._highlight_holds_from_endpoints(endpoints),
            contacts=self.contacts,
            step_count=self.step_count,
            ascent=float(self.player.pelvis[1] - self.initial_y),
            muscle_fatigue=self.muscle_model.state,
        )
        return self.renderer.render(ctx, self.render_mode)

    def close(self):
        self.renderer.close()

    def _build_obs(self) -> np.ndarray:
        endpoints = self.player.limb_endpoints()
        obs = []

        # Pelvis state (5)
        obs.extend([
            float((self.player.pelvis[0] / self.config.wall_width) * 2.0 - 1.0),
            float(self.player.pelvis[1] * 0.1),
            float(self.player.pelvis_velocity[0] * 0.3),
            float(self.player.pelvis_velocity[1] * 0.3),
            float((self.player.pelvis[1] - self.initial_y) * 0.1),
        ])

        # Joint angles (10): normalized to [-1, 1] within limits
        for name in JOINT_ORDER:
            lo, hi = JOINT_LIMITS[name]
            normalized = ((self.player.joint_angles[name] - lo) / (hi - lo)) * 2.0 - 1.0
            obs.append(float(normalized))
        # Joint velocities (10)
        for name in JOINT_ORDER:
            obs.append(float(np.tanh(self.player.joint_velocities[name] * 0.5)))

        # Contact flags (4) + contact positions relative to pelvis (8)
        obs.extend(self.contacts.as_array().tolist())
        obs.extend(self.contacts.as_position_obs(self.player.pelvis).tolist())

        # Per-limb hold obs (4 × visible_holds_per_limb × 3)
        # Each hold: (dx, dy) relative to limb endpoint + in_reach flag
        limb_endpoints = [
            endpoints.left_hand,
            endpoints.right_hand,
            endpoints.left_foot,
            endpoints.right_foot,
        ]
        for endpoint in limb_endpoints:
            ex, ey = float(endpoint[0]), float(endpoint[1])
            nearby = self.holds.nearest_holds(ex, ey, k=self.visible_holds_per_limb)
            for hold in nearby:
                dx = hold.x - ex
                dy = hold.y - ey
                dist = (dx * dx + dy * dy) ** 0.5
                in_reach = 1.0 if dist <= self.player_config.attach_radius else 0.0
                obs.extend([dx * 0.5, dy * 0.5, in_reach])
            # Pad if fewer than visible_holds_per_limb holds found
            for _ in range(self.visible_holds_per_limb - len(nearby)):
                obs.extend([0.0, 0.0, 0.0])

        # Per-muscle fatigue (10): map [0, 1] → [-1, 1]
        for name in JOINT_ORDER:
            obs.append(float(self.muscle_model.state.fatigue[name] * 2.0 - 1.0))

        return np.asarray(obs, dtype=np.float32)

    def _reach_potential(self, endpoints: LimbEndpoints) -> float:
        """Negative sum of distances from each unattached limb to its nearest hold.

        Higher (less negative) = limbs are closer to holds.
        Used for potential-based shaping: reward = weight * (F(s') - F(s)).
        """
        limb_data = [
            (endpoints.left_hand,  self.contacts.state.left_hand),
            (endpoints.right_hand, self.contacts.state.right_hand),
            (endpoints.left_foot,  self.contacts.state.left_foot),
            (endpoints.right_foot, self.contacts.state.right_foot),
        ]
        total = 0.0
        for ep, contact in limb_data:
            if contact is None:
                nearest = self.holds.nearest_holds(float(ep[0]), float(ep[1]), k=1)
                if nearest:
                    h = nearest[0]
                    dist = float(((h.x - ep[0]) ** 2 + (h.y - ep[1]) ** 2) ** 0.5)
                    total -= dist
        return total

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

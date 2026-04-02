from __future__ import annotations

import numpy as np

from bouldering2d.config import EnvConfig
from bouldering2d.environment.contacts import ContactManager
from bouldering2d.environment.player_model import PlayerModel


class PhysicsEngine:
    """Speed-oriented kinematics update with persistent gravity."""

    def __init__(self, config: EnvConfig) -> None:
        self.config = config

    def step(self, player: PlayerModel, contacts: ContactManager, torque_effort: float) -> tuple[float, bool]:
        prev_y = float(player.pelvis[1])
        support = contacts.state.attached_count()
        support_center = contacts.support_center(player.pelvis)

        # Temporary strength boost for easier climbing/debug iteration.
        up_pull = np.array([0.0, 0.0], dtype=np.float32)
        if support > 0:
            direction = support_center - player.pelvis
            up_pull = direction * (3.8 + 0.95 * support)

        gravity_force = np.array([0.0, -self.config.gravity], dtype=np.float32)
        control_lift = np.array([0.0, min(34.0, torque_effort * 0.13)], dtype=np.float32)
        support_damping = np.array([0.0, 18.0], dtype=np.float32) if support > 0 else np.zeros(2, dtype=np.float32)
        grip_boost = np.array([0.0, 5.0 * max(0, support - 1)], dtype=np.float32)
        lateral_pull = np.array([float(support_center[0] - player.pelvis[0]) * 2.0, 0.0], dtype=np.float32)

        acceleration = gravity_force + up_pull + control_lift + support_damping + grip_boost + lateral_pull
        player.update_pelvis(acceleration=acceleration, dt=self.config.dt)

        floor_y = -2.0
        fell = False
        if player.pelvis[1] < floor_y:
            player.pelvis[1] = floor_y
            player.pelvis_velocity[:] = 0.0
            fell = True

        return float(player.pelvis[1] - prev_y), fell

from __future__ import annotations

import numpy as np

from bouldering2d.config import EnvConfig
from bouldering2d.environment.contacts import ContactManager
from bouldering2d.environment.player_model import LimbEndpoints, PlayerModel


class PhysicsEngine:
    """Constraint-based climbing physics.

    The pelvis moves under gravity.  When limbs are attached to holds, a spring
    pulls the pelvis toward the position implied by the kinematic chain:

        pelvis_target_i = hold_i - (endpoint_i - pelvis)

    Bending a limb whose endpoint is attached to an overhead hold shortens the
    relative offset, raising the constraint target and pulling the pelvis up.
    Extending the limb lowers the target.  Random joint movements produce no
    net upward bias — the agent must earn height through correct body mechanics.
    """

    def __init__(self, config: EnvConfig) -> None:
        self.config = config

    def step(
        self,
        player: PlayerModel,
        contacts: ContactManager,
        endpoints: LimbEndpoints,
    ) -> tuple[float, bool]:
        prev_y = float(player.pelvis[1])

        gravity = np.array([0.0, -self.config.gravity], dtype=np.float32)

        # Kinematic constraint: compute where the pelvis must be so that each
        # attached limb endpoint coincides with its hold.
        targets: list[np.ndarray] = []
        limb_map = [
            (contacts.state.left_hand,  endpoints.left_hand),
            (contacts.state.right_hand, endpoints.right_hand),
            (contacts.state.left_foot,  endpoints.left_foot),
            (contacts.state.right_foot, endpoints.right_foot),
        ]
        for hold, endpoint in limb_map:
            if hold is not None:
                rel_offset = endpoint - player.pelvis
                hold_pos = np.array([hold.x, hold.y], dtype=np.float32)
                targets.append(hold_pos - rel_offset)

        if targets:
            pelvis_target = np.mean(targets, axis=0).astype(np.float32)
            spring_force = (pelvis_target - player.pelvis) * self.config.constraint_spring
            damping_force = -player.pelvis_velocity * self.config.constraint_damping
            acceleration = gravity + spring_force + damping_force
        else:
            acceleration = gravity

        player.update_pelvis(acceleration=acceleration, dt=self.config.dt)

        fell = False
        if player.pelvis[1] < -2.0:
            player.pelvis[1] = -2.0
            player.pelvis_velocity[:] = 0.0
            fell = True

        return float(player.pelvis[1] - prev_y), fell

from __future__ import annotations

import numpy as np

from bouldering2d.config import PlayerConfig
from bouldering2d.environment.player_model import JOINT_LIMITS, JOINT_ORDER, PlayerModel


def test_joint_limits_are_enforced() -> None:
    player = PlayerModel(PlayerConfig())
    large_actions = np.ones(len(JOINT_ORDER), dtype=np.float32)

    for _ in range(200):
        player.apply_joint_actions(large_actions, dt=1.0 / 30.0)

    for name in JOINT_ORDER:
        lo, hi = JOINT_LIMITS[name]
        assert lo <= player.joint_angles[name] <= hi

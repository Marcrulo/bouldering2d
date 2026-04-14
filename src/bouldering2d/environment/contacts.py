from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bouldering2d.config import PlayerConfig
from bouldering2d.environment.holds import Hold, HoldField
from bouldering2d.environment.player_model import LimbEndpoints


@dataclass(slots=True)
class ContactState:
    left_hand: Hold | None = None
    right_hand: Hold | None = None
    left_foot: Hold | None = None
    right_foot: Hold | None = None

    def attached_count(self) -> int:
        return sum(int(v is not None) for v in [self.left_hand, self.right_hand, self.left_foot, self.right_foot])


class ContactManager:
    def __init__(self, player_config: PlayerConfig) -> None:
        self.player_config = player_config
        self.state = ContactState()

    def reset(self) -> None:
        self.state = ContactState()

    def update(self, grip_actions: np.ndarray, endpoints: LimbEndpoints, holds: HoldField) -> int:
        new_contacts = 0
        attach_distance = self.player_config.attach_radius
        release_distance = self.player_config.contact_release_dist
        grip_map = {
            "left_hand": (float(grip_actions[0]), endpoints.left_hand),
            "right_hand": (float(grip_actions[1]), endpoints.right_hand),
            "left_foot": (float(grip_actions[2]), endpoints.left_foot),
            "right_foot": (float(grip_actions[3]), endpoints.right_foot),
        }

        for limb, (cmd, point) in grip_map.items():
            if cmd < -0.3:
                setattr(self.state, limb, None)
                continue

            current = getattr(self.state, limb)
            if current is not None:
                if np.linalg.norm(point - np.array([current.x, current.y], dtype=np.float32)) > release_distance:
                    setattr(self.state, limb, None)
                continue

            if cmd <= 0.3:
                continue

            nearest = self._find_reachable_hold(point, holds, attach_distance)
            if nearest is not None:
                setattr(self.state, limb, nearest)
                new_contacts += 1

        return new_contacts

    def support_center(self, fallback: np.ndarray) -> np.ndarray:
        pts = []
        for hold in [
            self.state.left_hand,
            self.state.right_hand,
            self.state.left_foot,
            self.state.right_foot,
        ]:
            if hold is not None:
                pts.append([hold.x, hold.y])
        if not pts:
            return fallback
        return np.mean(np.asarray(pts, dtype=np.float32), axis=0)

    def as_array(self) -> np.ndarray:
        return np.array(
            [
                1.0 if self.state.left_hand else 0.0,
                1.0 if self.state.right_hand else 0.0,
                1.0 if self.state.left_foot else 0.0,
                1.0 if self.state.right_foot else 0.0,
            ],
            dtype=np.float32,
        )

    def as_position_obs(self, pelvis: np.ndarray) -> np.ndarray:
        """Return the hold position (relative to pelvis) for each limb, or (0,0) if not attached.

        Gives the agent spatial awareness of where its current contacts are,
        so it can learn to release holds that are far below.
        """
        vals: list[float] = []
        for hold in [self.state.left_hand, self.state.right_hand, self.state.left_foot, self.state.right_foot]:
            if hold is not None:
                vals.append(float((hold.x - pelvis[0]) * 0.6))
                vals.append(float((hold.y - pelvis[1]) * 0.4))
            else:
                vals.append(0.0)
                vals.append(0.0)
        return np.array(vals, dtype=np.float32)

    def stale_count(self, pelvis_y: float, threshold: float = -0.8) -> int:
        """Count contacts whose hold is more than `threshold` metres below the pelvis.

        Used for a penalty signal to discourage holding onto holds that are
        far below the current body position.
        """
        count = 0
        for hold in [self.state.left_hand, self.state.right_hand, self.state.left_foot, self.state.right_foot]:
            if hold is not None and (hold.y - pelvis_y) < threshold:
                count += 1
        return count

    def _find_reachable_hold(self, point: np.ndarray, holds: HoldField, attach_distance: float) -> Hold | None:
        nearby = holds.nearest_holds(float(point[0]), float(point[1]), k=10)
        for h in nearby:
            if np.linalg.norm(point - np.array([h.x, h.y], dtype=np.float32)) <= attach_distance:
                return h
        return None

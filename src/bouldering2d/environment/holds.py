from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bouldering2d.config import EnvConfig


@dataclass(slots=True)
class Hold:
    x: float
    y: float


class HoldField:
    """Maintains a rolling set of holds around the climber height."""

    def __init__(self, config: EnvConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self._holds: list[Hold] = []
        self._max_generated_y = 0.0

    def reset(self, player_y: float) -> None:
        self._holds.clear()
        self._max_generated_y = player_y - 2.0
        self._seed_start_cluster(player_y)
        self._ensure_coverage(player_y - 3.0, player_y + 10.0)

    def _seed_start_cluster(self, player_y: float) -> None:
        cx = self.config.wall_width * 0.5
        # Offsets derived from the rest-pose limb endpoints (pelvis at player_y):
        #   hands  ≈ (±0.65, +1.58) from pelvis — arms outstretched, slightly up
        #   feet   ≈ (±0.00, -1.00) from pelvis — legs hanging down together
        # Holds are placed at each endpoint so the first contacts.update() attaches
        # immediately and the constraint spring starts from a neutral (zero-force) state.
        # A second tier of hand holds sits ~0.6m above for the first reach.
        # Offsets from rest-pose kinematics (pelvis at player_y, x=wall_width/2):
        #   left_hand  ≈ (+0.45, +1.58),  right_hand ≈ (-0.58, +1.58)
        #   both feet  ≈ ( 0.00, -0.99)
        anchors = [
            # Hand holds exactly at rest-pose hand positions
            (cx + 0.45, player_y + 1.58),   # left hand
            (cx - 0.58, player_y + 1.58),   # right hand
            # First reach — above for the agent's initial pull
            (cx + 0.45, player_y + 2.20),
            (cx - 0.58, player_y + 2.20),
            # Foot holds exactly at rest-pose foot positions
            (cx + 0.05, player_y - 0.99),
            (cx - 0.05, player_y - 0.99),
        ]
        for x, y in anchors:
            x = float(np.clip(x, 0.3, self.config.wall_width - 0.3))
            self._holds.append(Hold(x=x, y=float(y)))

    def _ensure_coverage(self, y_min: float, y_max: float) -> None:
        if self._max_generated_y < y_min:
            self._max_generated_y = y_min

        while self._max_generated_y <= y_max and len(self._holds) < self.config.max_holds:
            self._max_generated_y += self.config.hold_spacing_y
            num_holds = int(self.rng.integers(3, 6))
            for _ in range(num_holds):
                cx = self.config.wall_width * 0.5
                x = float(
                    np.clip(
                        cx + self.rng.uniform(-2.8, 2.8) + self.rng.normal(0.0, self.config.hold_jitter_x),
                        0.3,
                        self.config.wall_width - 0.3,
                    )
                )
                y = float(self._max_generated_y + self.rng.normal(0.0, 0.12))
                self._holds.append(Hold(x=x, y=y))

        self._holds = [h for h in self._holds if h.y >= y_min - 2.0]

    def visible_holds(self, player_y: float) -> list[Hold]:
        self._ensure_coverage(player_y - 6.0, player_y + 14.0)
        low = player_y - 6.0
        high = player_y + 14.0
        return [h for h in self._holds if low <= h.y <= high]

    def nearest_holds(self, x: float, y: float, k: int = 12, max_dist: float | None = None) -> list[Hold]:
        if not self._holds:
            return []
        d2 = [(h, (h.x - x) ** 2 + (h.y - y) ** 2) for h in self._holds]
        if max_dist is not None:
            max_d2 = max_dist * max_dist
            d2 = [(h, dist) for h, dist in d2 if dist <= max_d2]
        d2.sort(key=lambda item: item[1])
        return [h for h, _ in d2[:k]]

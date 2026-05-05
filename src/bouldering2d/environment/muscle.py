from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bouldering2d.config import MuscleConfig
from bouldering2d.environment.player_model import JOINT_ORDER


@dataclass(slots=True)
class MuscleState:
    """Per-muscle fatigue in [0.0, 1.0]. 0 = fresh, 1 = exhausted."""
    fatigue: dict

    def as_array(self) -> np.ndarray:
        return np.array([self.fatigue[j] for j in JOINT_ORDER], dtype=np.float32)


@dataclass(slots=True)
class JointActionResult:
    joint_torques: dict   # effective torque per joint (post-fatigue)
    total_effort: float   # sum of |effective_torques|


class MuscleModel:
    def __init__(self, config: MuscleConfig) -> None:
        self.config = config
        self.state = MuscleState(fatigue={j: 0.0 for j in JOINT_ORDER})

    def reset(self) -> None:
        for j in JOINT_ORDER:
            self.state.fatigue[j] = 0.0

    def angle_multiplier(self, joint: str, angle: float) -> float:
        """
        Gaussian torque-angle curve: peak at optimal_angle, falls off toward extremes.
        Returns a value in [angle_min_strength, 1.0].
        """
        optimal = self.config.optimal_angle[joint]
        width = self.config.angle_width[joint]
        raw = float(np.exp(-0.5 * ((angle - optimal) / width) ** 2))
        mn = self.config.angle_min_strength
        return mn + (1.0 - mn) * raw

    def effective_torque_scale(self, joint: str, angle: float) -> float:
        """Max torque scale reduced by fatigue and current joint angle."""
        base = self.config.torque_scales[joint]
        angle_mult = self.angle_multiplier(joint, angle)
        fatigue_penalty = self.config.fatigue_penalty_scale * self.state.fatigue[joint]
        return base * angle_mult * (1.0 - fatigue_penalty)

    def step(self, joint_torques: dict, dt: float) -> None:
        for j in JOINT_ORDER:
            torque_mag = abs(joint_torques[j])
            base_scale = self.config.torque_scales[j]
            normalized_effort = torque_mag / base_scale if base_scale > 0 else 0.0
            delta = self.config.fatigue_rate[j] * normalized_effort * dt
            self.state.fatigue[j] = min(1.0, self.state.fatigue[j] + delta)

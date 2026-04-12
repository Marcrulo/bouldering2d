from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bouldering2d.config import PlayerConfig


JOINT_ORDER = [
    "neck",
    "spine",
    "l_shoulder",
    "r_shoulder",
    "l_elbow",
    "r_elbow",
    "l_hip",
    "r_hip",
    "l_knee",
    "r_knee",
]

JOINT_LIMITS = {
    "neck": (-0.6, 0.6),
    "spine": (-0.5, 0.6),
    "l_shoulder": (-2.6, 0.0),
    "r_shoulder": (0.0, 2.6),
    "l_elbow": (-2.6, -0.1),
    "r_elbow": (0.1, 2.6),
    "l_hip": (-1.5, -0.3),
    "r_hip": (0.3, 1.5),
    "l_knee": (0.0, 2.0),
    "r_knee": (0.0, 2.0),
}


@dataclass(slots=True)
class LimbEndpoints:
    head: np.ndarray
    upper_torso: np.ndarray
    mid_torso: np.ndarray
    lower_torso: np.ndarray
    left_elbow: np.ndarray
    right_elbow: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    left_knee: np.ndarray
    right_knee: np.ndarray
    left_foot: np.ndarray
    right_foot: np.ndarray


class PlayerModel:
    """Simplified articulated climber model with anatomically clamped joints."""

    def __init__(self, config: PlayerConfig) -> None:
        self.config = config
        self.pelvis = np.array([4.0, 0.8], dtype=np.float32)
        self.pelvis_velocity = np.zeros(2, dtype=np.float32)
        self.joint_angles = {name: 0.0 for name in JOINT_ORDER}
        self.joint_velocities = {name: 0.0 for name in JOINT_ORDER}
        self._set_rest_pose()

    def _set_rest_pose(self) -> None:
        self.joint_angles["spine"] = 0.05
        self.joint_angles["neck"] = 0.0
        self.joint_angles["l_shoulder"] = -1.4
        self.joint_angles["r_shoulder"] = 1.4
        self.joint_angles["l_elbow"] = -1.2
        self.joint_angles["r_elbow"] = 1.2
        self.joint_angles["l_hip"] = -0.3
        self.joint_angles["r_hip"] = 0.3
        self.joint_angles["l_knee"] = 0.6
        self.joint_angles["r_knee"] = 0.6

    def reset(self, x: float = 4.0, y: float = 0.8) -> None:
        self.pelvis[:] = (x, y)
        self.pelvis_velocity[:] = (0.0, 0.0)
        self.joint_velocities = {name: 0.0 for name in JOINT_ORDER}
        self._set_rest_pose()

    def apply_joint_actions(self, actions: np.ndarray, dt: float) -> float:
        efforts = 0.0
        for idx, name in enumerate(JOINT_ORDER):
            torque = float(actions[idx]) * self.config.joint_torque_scale
            self.joint_velocities[name] = (
                self.joint_velocities[name] * self.config.joint_damping + torque * dt
            )
            self.joint_angles[name] += self.joint_velocities[name] * dt
            lo, hi = JOINT_LIMITS[name]
            if self.joint_angles[name] < lo:
                self.joint_angles[name] = lo
                self.joint_velocities[name] = 0.0
            elif self.joint_angles[name] > hi:
                self.joint_angles[name] = hi
                self.joint_velocities[name] = 0.0
            efforts += abs(torque)
        return efforts

    def update_pelvis(self, acceleration: np.ndarray, dt: float) -> None:
        self.pelvis_velocity += acceleration.astype(np.float32) * dt
        self.pelvis_velocity *= self.config.pelvis_drag
        self.pelvis += self.pelvis_velocity * dt

    def limb_endpoints(self) -> LimbEndpoints:
        px, py = float(self.pelvis[0]), float(self.pelvis[1])
        pelvis = np.array([px, py], dtype=np.float32)

        lower_torso_angle = np.pi / 2 + self.joint_angles["spine"]
        upper_torso_angle = lower_torso_angle + self.joint_angles["spine"] * 0.6
        chest = pelvis + self._vec(lower_torso_angle, self.config.torso_lower_length)
        neck_base = chest + self._vec(upper_torso_angle, self.config.torso_upper_length)
        head = neck_base + self._vec(upper_torso_angle + self.joint_angles["neck"], self.config.neck_length)

        l_upper_arm_angle = np.pi / 2 + self.joint_angles["l_shoulder"]
        r_upper_arm_angle = np.pi / 2 + self.joint_angles["r_shoulder"]
        l_elbow = neck_base + self._vec(l_upper_arm_angle, self.config.upper_arm_length)
        r_elbow = neck_base + self._vec(r_upper_arm_angle, self.config.upper_arm_length)
        l_forearm_angle = l_upper_arm_angle - self.joint_angles["l_elbow"]
        r_forearm_angle = r_upper_arm_angle - self.joint_angles["r_elbow"]
        l_wrist = l_elbow + self._vec(l_forearm_angle, self.config.lower_arm_length)
        r_wrist = r_elbow + self._vec(r_forearm_angle, self.config.lower_arm_length)
        l_hand = l_wrist + self._vec(l_forearm_angle, self.config.hand_length)
        r_hand = r_wrist + self._vec(r_forearm_angle, self.config.hand_length)

        l_thigh_angle = -np.pi / 2 + self.joint_angles["l_hip"]
        r_thigh_angle = -np.pi / 2 + self.joint_angles["r_hip"]
        l_knee = pelvis + self._vec(l_thigh_angle, self.config.upper_leg_length)
        r_knee = pelvis + self._vec(r_thigh_angle, self.config.upper_leg_length)
        l_shin_angle = l_thigh_angle + self.joint_angles["l_knee"]
        r_shin_angle = r_thigh_angle - self.joint_angles["r_knee"]
        l_foot = l_knee + self._vec(l_shin_angle, self.config.lower_leg_length)
        r_foot = r_knee + self._vec(r_shin_angle, self.config.lower_leg_length)

        return LimbEndpoints(
            head=head,
            upper_torso=neck_base,
            mid_torso=chest,
            lower_torso=pelvis,
            left_elbow=l_elbow,
            right_elbow=r_elbow,
            left_hand=l_hand,
            right_hand=r_hand,
            left_knee=l_knee,
            right_knee=r_knee,
            left_foot=l_foot,
            right_foot=r_foot,
        )

    @staticmethod
    def _vec(angle: float, length: float) -> np.ndarray:
        return np.array([np.cos(angle) * length, np.sin(angle) * length], dtype=np.float32)

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EnvConfig:
    screen_width: int = 900
    screen_height: int = 1200
    wall_width: float = 8.0
    max_holds: int = 1800
    hold_spacing_y: float = 0.35
    hold_jitter_x: float = 0.8
    hold_size: float = 0.16
    gravity: float = 9.81
    dt: float = 1.0 / 30.0
    max_steps: int = 1200
    start_stamina: float = 100.0
    min_stamina: float = 0.0
    ascent_target: float = 20.0
    seed: int = 42
    constraint_spring: float = 100.0


@dataclass(slots=True)
class PlayerConfig:
    torso_lower_length: float = 0.55
    torso_upper_length: float = 0.45
    neck_length: float = 0.15
    upper_arm_length: float = 0.42
    lower_arm_length: float = 0.42
    hand_length: float = 0.10
    upper_leg_length: float = 0.52
    lower_leg_length: float = 0.52
    joint_torque_scale: float = 5.0
    joint_damping: float = 0.92
    pelvis_drag: float = 0.92
    attach_radius: float = 0.28
    contact_release_dist: float = 2.0


@dataclass(slots=True)
class MuscleConfig:
    torque_scales: dict = field(default_factory=lambda: {
        "neck":       1.5,
        "spine":      4.0,
        "l_shoulder": 3.5,  "r_shoulder": 3.5,
        "l_elbow":    3.0,  "r_elbow":    3.0,
        "l_hip":      7.0,  "r_hip":      7.0,
        "l_knee":     6.5,  "r_knee":     6.5,
    })
    fatigue_rate: dict = field(default_factory=lambda: {
        "neck":       0.04,
        "spine":      0.05,
        "l_shoulder": 0.10,  "r_shoulder": 0.10,
        "l_elbow":    0.14,  "r_elbow":    0.14,
        "l_hip":      0.04,  "r_hip":      0.04,
        "l_knee":     0.03,  "r_knee":     0.03,
    })
    recovery_rate: dict = field(default_factory=lambda: {
        "neck":       0.20,
        "spine":      0.15,
        "l_shoulder": 0.10,  "r_shoulder": 0.10,
        "l_elbow":    0.08,  "r_elbow":    0.08,
        "l_hip":      0.18,  "r_hip":      0.18,
        "l_knee":     0.20,  "r_knee":     0.20,
    })
    rest_threshold: float = 0.15
    fatigue_penalty_scale: float = 0.60
    # Angle where each muscle produces peak torque (radians)
    optimal_angle: dict = field(default_factory=lambda: {
        "neck":       0.0,
        "spine":      0.05,
        "l_shoulder": -1.3,  "r_shoulder": 1.3,
        "l_elbow":    -1.6,  "r_elbow":    1.6,
        "l_hip":      -0.9,  "r_hip":      0.9,
        "l_knee":     1.0,   "r_knee":     1.0,
    })
    # Width of the Gaussian strength curve (larger = strength holds over wider range)
    angle_width: dict = field(default_factory=lambda: {
        "neck":       0.50,
        "spine":      0.50,
        "l_shoulder": 0.80,  "r_shoulder": 0.80,
        "l_elbow":    0.70,  "r_elbow":    0.70,
        "l_hip":      0.70,  "r_hip":      0.70,
        "l_knee":     0.70,  "r_knee":     0.70,
    })
    # Floor: muscle retains at least this fraction of max torque at any angle
    angle_min_strength: float = 0.25


@dataclass(slots=True)
class StaminaConfig:
    base_drain_per_sec: float = 0.65
    movement_coeff: float = 0.09
    tension_coeff: float = 0.06
    fall_penalty: float = 8.0


@dataclass(slots=True)
class RewardConfig:
    ascent_weight: float = 6.0
    energy_penalty_weight: float = 0.02
    fall_penalty: float = 25.0
    contact_bonus: float = 0.03
    idle_penalty: float = 0.002
    stale_contact_penalty: float = 0.005  # per stale contact per step


@dataclass(slots=True)
class TrainingConfig:
    total_timesteps: int = 300_000
    eval_interval: int = 10_000
    eval_episodes: int = 2
    checkpoint_interval: int = 25_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 128
    gamma: float = 0.99
    tensorboard_log_dir: str = "data/logs"
    checkpoint_dir: str = "data/checkpoints"
    video_dir: str = "data/videos"
    seed: int = 42
    policy_kwargs: dict = field(default_factory=lambda: {"net_arch": [256, 256]})

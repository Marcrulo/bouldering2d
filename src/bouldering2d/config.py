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
    foot_length: float = 0.15
    joint_torque_scale: float = 5.0
    joint_damping: float = 0.92
    pelvis_drag: float = 0.85
    attach_radius: float = 0.22


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

from __future__ import annotations

from dataclasses import dataclass

from bouldering2d.config import RewardConfig


@dataclass(slots=True)
class RewardBreakdown:
    total: float
    ascent: float
    energy: float
    contact: float
    idle: float
    fall: float


class RewardModel:
    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def compute(
        self,
        dy: float,
        stamina_spent: float,
        new_contacts: int,
        attachments: int,
        fell: bool,
        stale_contacts: int = 0,
    ) -> RewardBreakdown:
        ascent = dy * self.config.ascent_weight
        energy = -stamina_spent * self.config.energy_penalty_weight
        contact = new_contacts * self.config.contact_bonus - stale_contacts * self.config.stale_contact_penalty
        idle = -self.config.idle_penalty if abs(dy) < 0.001 and attachments > 0 else 0.0
        fall = -self.config.fall_penalty if fell else 0.0
        total = ascent + energy + contact + idle + fall
        return RewardBreakdown(total=total, ascent=ascent, energy=energy, contact=contact, idle=idle, fall=fall)

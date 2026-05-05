from __future__ import annotations

from dataclasses import dataclass

from bouldering2d.config import RewardConfig


@dataclass(slots=True)
class RewardBreakdown:
    total: float
    ascent: float
    energy: float = 0.0
    contact: float = 0.0
    idle: float = 0.0
    fall: float = 0.0
    reach: float = 0.0
    stale: float = 0.0


class RewardModel:
    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    def compute(
        self,
        dy: float,
        stamina_spent: float = 0.0,
        new_contacts: int = 0,
        attachments: int = 0,
        fell: bool = False,
        stale_contacts: int = 0,
        reach_delta: float = 0.0,
    ) -> RewardBreakdown:
        cfg = self.config
        ascent  =  cfg.ascent_weight * dy
        energy  = -cfg.energy_penalty_weight * stamina_spent
        # small one-time signal for discovering a hold + continuous bonus for holding it
        contact =  cfg.contact_bonus * new_contacts + cfg.holding_bonus * attachments
        fall    = -cfg.fall_penalty * (1.0 if fell else 0.0)
        idle    = -cfg.idle_penalty * (1.0 if attachments == 0 else 0.0)
        stale   = -cfg.stale_contact_penalty * stale_contacts
        reach   =  reach_delta  # already scaled by reach_shaping_weight in env

        total = ascent + energy + contact + fall + idle + stale + reach
        return RewardBreakdown(
            total=total,
            ascent=ascent,
            energy=energy,
            contact=contact,
            idle=idle,
            fall=fall,
            reach=reach,
            stale=stale,
        )

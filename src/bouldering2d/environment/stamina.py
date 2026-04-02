from __future__ import annotations

from dataclasses import dataclass

from bouldering2d.config import EnvConfig, StaminaConfig


@dataclass(slots=True)
class StaminaResult:
    stamina: float
    spent: float


class StaminaModel:
    def __init__(self, env_config: EnvConfig, stamina_config: StaminaConfig) -> None:
        self.env_config = env_config
        self.config = stamina_config

    def update(self, stamina: float, effort: float, attachments: int, dt: float) -> StaminaResult:
        support_gap = max(0.0, 2.0 - attachments)
        drain = (
            self.config.base_drain_per_sec
            + effort * self.config.movement_coeff
            + support_gap * self.config.tension_coeff * 10.0
        ) * dt
        next_stamina = max(self.env_config.min_stamina, stamina - drain)
        return StaminaResult(stamina=next_stamina, spent=drain)

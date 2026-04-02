from __future__ import annotations

from bouldering2d.config import EnvConfig, StaminaConfig
from bouldering2d.environment.stamina import StaminaModel


def test_stamina_decreases_with_effort() -> None:
    model = StaminaModel(EnvConfig(), StaminaConfig())
    low = model.update(stamina=100.0, effort=0.1, attachments=2, dt=1 / 30)
    high = model.update(stamina=100.0, effort=10.0, attachments=0, dt=1 / 30)
    assert high.stamina < low.stamina


def test_stamina_not_negative() -> None:
    model = StaminaModel(EnvConfig(), StaminaConfig())
    result = model.update(stamina=0.01, effort=100.0, attachments=0, dt=1.0)
    assert result.stamina >= 0.0

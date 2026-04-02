from __future__ import annotations

from bouldering2d.config import RewardConfig
from bouldering2d.environment.reward import RewardModel


def test_ascent_reward_positive() -> None:
    model = RewardModel(RewardConfig())
    result = model.compute(dy=0.2, stamina_spent=0.1, new_contacts=0, attachments=2, fell=False)
    assert result.total > 0.0


def test_fall_is_penalized() -> None:
    model = RewardModel(RewardConfig())
    safe = model.compute(dy=0.0, stamina_spent=0.0, new_contacts=0, attachments=0, fell=False)
    fall = model.compute(dy=0.0, stamina_spent=0.0, new_contacts=0, attachments=0, fell=True)
    assert fall.total < safe.total

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from bouldering2d.environment.climbing_env import BoulderingEnv


def test_env_checker() -> None:
    env = BoulderingEnv(render_mode=None)
    check_env(env, skip_render_check=True)
    env.close()


def test_observation_shape_and_dtype() -> None:
    env = BoulderingEnv(render_mode=None)
    obs, _ = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "ascent" in info
    env.close()

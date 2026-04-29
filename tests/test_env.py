import numpy as np

from loba_ai.env import LobaEnv


def test_env_reset_and_step_smoke():
    env = LobaEnv(seed=7)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert "action_mask" in info

    action = int(np.flatnonzero(info["action_mask"])[0])
    obs2, reward, terminated, truncated, info2 = env.step(action)
    assert obs2.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "action_mask" in info2

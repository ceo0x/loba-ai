import numpy as np
from gymnasium import spaces

from loba_ai.model_io import choose_action


class _DummyModel:
    def __init__(self, expected_obs_dim: int) -> None:
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(expected_obs_dim,), dtype=np.float32)
        self.seen_obs_shape: tuple[int, ...] | None = None

    def predict(self, obs, deterministic=True, action_masks=None):
        self.seen_obs_shape = tuple(obs.shape)
        valid = np.flatnonzero(action_masks)
        return int(valid[0]), None


def test_choose_action_adapts_115_to_122():
    model = _DummyModel(expected_obs_dim=122)
    obs = np.zeros(115, dtype=np.float32)
    mask = np.zeros(36, dtype=bool)
    mask[3] = True
    action = choose_action(model, obs, mask)
    assert action == 3
    assert model.seen_obs_shape == (122,)


def test_choose_action_raises_for_unsupported_mapping():
    model = _DummyModel(expected_obs_dim=130)
    obs = np.zeros(115, dtype=np.float32)
    mask = np.zeros(36, dtype=bool)
    mask[0] = True
    try:
        choose_action(model, obs, mask)
        assert False, "Expected ValueError for unsupported obs mapping"
    except ValueError as exc:
        assert "Model observation mismatch" in str(exc)

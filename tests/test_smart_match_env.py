import numpy as np

from loba_ai.rules import Rules
from loba_ai.smart_match_env import SmartMatchLobaEnv


def test_smart_match_env_observation_shape_extends_base_match_obs():
    env = SmartMatchLobaEnv(rules=Rules(num_players=2), target_points=100)
    obs, info = env.reset(seed=7)
    assert obs.shape == (225,)
    assert "action_mask" in info


def test_smart_match_env_seen_features_are_normalized():
    env = SmartMatchLobaEnv(rules=Rules(num_players=2), target_points=100)
    obs, _ = env.reset(seed=13)
    # Last 107 features are: seen_ratio(53), remaining_ratio(53), seen_total(1)
    smart_tail = obs[-107:]
    assert np.all(smart_tail >= 0.0)
    assert np.all(smart_tail <= 1.0)
    # At reset we should have seen at least own hand + top discard.
    assert smart_tail[-1] > 0.0


def test_smart_match_env_does_not_double_count_same_visible_cards():
    env = SmartMatchLobaEnv(rules=Rules(num_players=2), target_points=100)
    env.reset(seed=23)
    before = env._seen_token_counts.copy()
    env._update_seen_from_state()
    after = env._seen_token_counts.copy()
    assert np.array_equal(before, after)

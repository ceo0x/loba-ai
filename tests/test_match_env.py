import numpy as np
import pytest
from gymnasium import spaces

from loba_ai.cards import Card
from loba_ai.match_env import MatchLobaEnv
from loba_ai.rules import Rules
from loba_ai.state import GameState, PlayerState


def _build_round_end_state(p0_hand, p1_hand, winner: int = 0) -> GameState:
    p0 = PlayerState(hand=list(p0_hand))
    p1 = PlayerState(hand=list(p1_hand))
    return GameState(
        players=[p0, p1],
        current_player=winner,
        stock_pile=[Card(rank=9, suit="clubs", deck_id=0)],
        discard_pile=[Card(rank=8, suit="spades", deck_id=0)],
        melds_on_table=[],
        round_index=0,
        turn_number=12,
        finished=True,
        winner=winner,
        phase="discard",
    )


def test_match_env_reset_shape_and_mask():
    env = MatchLobaEnv(rules=Rules(num_players=2), target_points=100)
    obs, info = env.reset(seed=123)
    assert obs.shape == (118,)
    assert "action_mask" in info
    assert len(info["action_mask"]) == env.action_space.n
    assert info["match_scores"] == [0.0, 0.0]


def test_finish_round_accumulates_scores_and_continues_when_under_target():
    env = MatchLobaEnv(rules=Rules(num_players=2), target_points=100, max_rounds_per_match=10)
    env.reset(seed=1)
    env.engine.state = _build_round_end_state(
        p0_hand=[],
        p1_hand=[Card(rank=13, suit="hearts", deck_id=0), Card(rank=9, suit="clubs", deck_id=0)],
        winner=0,
    )

    _, done, info = env._finish_round()
    assert done is False
    assert env.rounds_played == 1
    assert np.allclose(env.match_scores, np.array([0.0, 19.0], dtype=np.float32))
    assert info["round_over"] is True
    assert env.engine.state.finished is False
    assert env.engine.state.phase == "draw"


def test_finish_round_terminates_match_when_target_reached():
    env = MatchLobaEnv(rules=Rules(num_players=2), target_points=20, max_rounds_per_match=10)
    env.reset(seed=2)
    env.match_scores = np.array([10.0, 19.0], dtype=np.float32)
    env.engine.state = _build_round_end_state(
        p0_hand=[Card(rank=5, suit="diamonds", deck_id=0)],
        p1_hand=[Card(rank=2, suit="clubs", deck_id=0)],
        winner=1,
    )

    _, done, info = env._finish_round()
    assert done is True
    assert info["match_finished"] is True
    assert info["match_winner"] == 0
    assert env.match_scores[1] >= 20.0


class _DummyModel:
    def __init__(self) -> None:
        self.called = False
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(122,), dtype=np.float32)

    def predict(self, obs, deterministic=True, action_masks=None):
        self.called = True
        valid = np.flatnonzero(action_masks)
        return int(valid[0]), None


def test_mixed4p_forces_four_players_and_dynamic_obs_shape():
    env = MatchLobaEnv(
        rules=Rules(num_players=2),
        table_mode="mixed4p",
        trained_opponent_model=_DummyModel(),
        target_points=100,
    )
    obs, info = env.reset(seed=5)
    assert env.rules.num_players == 4
    assert obs.shape == env.observation_space.shape
    assert obs.shape == (122,)
    assert len(info["match_scores"]) == 4


def test_mixed4p_uses_trained_model_on_player_one_turn():
    dummy = _DummyModel()
    env = MatchLobaEnv(
        table_mode="mixed4p",
        trained_opponent_model=dummy,
        target_points=100,
    )
    env.reset(seed=9)
    env.engine.state.current_player = 1
    mask = env._action_mask()
    _ = env._choose_opponent_action(env.engine.state.phase, mask, player_index=1)
    assert dummy.called is True


def test_mixed4p_can_adapt_obs_for_118d_trained_model():
    dummy = _DummyModel()
    dummy.observation_space = spaces.Box(low=0.0, high=1.0, shape=(118,), dtype=np.float32)
    env = MatchLobaEnv(
        table_mode="mixed4p",
        trained_opponent_model=dummy,
        target_points=100,
    )
    env.reset(seed=11)
    obs = env._obs_for_player_with_dim(player_index=1, expected_dim=118)
    assert obs.shape == (118,)


def test_finish_round_done_reports_joker_telemetry_keys():
    env = MatchLobaEnv(rules=Rules(num_players=2), target_points=1, max_rounds_per_match=2)
    env.reset(seed=21)
    env.joker_stats["melds_total"] = 4
    env.joker_stats["melds_with_joker"] = 1
    env.joker_stats["joker_discards"] = 2
    env.joker_stats["avoidable_joker_melds"] = 1
    env.joker_stats["agent_turns"] = 5
    env.joker_stats["agent_turns_holding_joker"] = 3
    env.match_scores = np.array([1.0, 2.0], dtype=np.float32)
    env.engine.state = _build_round_end_state(
        p0_hand=[],
        p1_hand=[Card(rank=2, suit="clubs", deck_id=0)],
        winner=0,
    )

    _, done, info = env._finish_round()
    assert done is True
    assert info["joker_meld_rate"] == pytest.approx(0.25)
    assert info["joker_hold_rate"] == pytest.approx(0.6)
    assert info["joker_discards"] == 2
    assert info["avoidable_joker_melds"] == 1

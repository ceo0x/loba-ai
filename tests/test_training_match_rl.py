import argparse

import pytest

from loba_ai.training.train_match_rl import (
    MatchRoundsLoggingCallback,
    _parse_curriculum_ratios,
    resolve_opponent_model_path,
)


def test_resolve_opponent_model_path_prefers_cli_value():
    args = argparse.Namespace(
        table_mode="mixed4p",
        opponent_model_path="artifacts/cli_model.zip",
        default_opponent_model_path="artifacts/default_model.zip",
    )
    assert resolve_opponent_model_path(args) == "artifacts/cli_model.zip"


def test_resolve_opponent_model_path_falls_back_to_default():
    args = argparse.Namespace(
        table_mode="mixed4p",
        opponent_model_path=None,
        default_opponent_model_path="artifacts/default_model.zip",
    )
    assert resolve_opponent_model_path(args) == "artifacts/default_model.zip"


def test_resolve_opponent_model_path_requires_value_in_mixed4p():
    args = argparse.Namespace(
        table_mode="mixed4p",
        opponent_model_path=None,
        default_opponent_model_path="",
    )
    with pytest.raises(ValueError):
        resolve_opponent_model_path(args)


def test_resolve_opponent_model_path_not_needed_in_classic():
    args = argparse.Namespace(
        table_mode="classic",
        opponent_model_path=None,
        default_opponent_model_path="",
    )
    assert resolve_opponent_model_path(args) is None


def test_match_rounds_logging_callback_records_finished_match_rounds():
    callback = MatchRoundsLoggingCallback(window_size=5)
    callback.locals = {
        "infos": [
            {
                "match_finished": True,
                "rounds_played": 7,
                "joker_meld_rate": 0.5,
                "joker_hold_rate": 0.6,
                "joker_discards": 2,
                "avoidable_joker_melds": 1,
            },
            {"match_finished": False, "rounds_played": 4},
            {
                "match_finished": True,
                "rounds_played": 9,
                "joker_meld_rate": 0.25,
                "joker_hold_rate": 0.8,
                "joker_discards": 1,
                "avoidable_joker_melds": 0,
            },
        ]
    }
    callback._on_step()
    assert list(callback._rounds_buffer) == [7.0, 9.0]
    assert list(callback._joker_meld_rate_buffer) == [0.5, 0.25]
    assert list(callback._joker_hold_rate_buffer) == [0.6, 0.8]


def test_match_rounds_logging_callback_writes_mean_on_rollout_end():
    callback = MatchRoundsLoggingCallback(window_size=5)
    callback._rounds_buffer.extend([6.0, 8.0, 10.0])
    callback._joker_meld_rate_buffer.extend([0.3, 0.4])
    callback._joker_hold_rate_buffer.extend([0.8, 0.6])
    callback._joker_discards_buffer.extend([2.0, 1.0])
    callback._avoidable_joker_melds_buffer.extend([1.0, 0.0])
    captured: dict[str, float] = {}

    class _Logger:
        def record(self, key: str, value: float) -> None:
            captured[key] = value

    class _Model:
        logger = _Logger()

    callback.init_callback(_Model())  # type: ignore[arg-type]
    callback._on_rollout_end()
    assert captured["train/ep_rounds_mean"] == 8.0
    assert captured["train/joker_meld_rate_mean"] == 0.35
    assert captured["train/joker_hold_rate_mean"] == 0.7
    assert captured["train/joker_discards_mean"] == 1.5
    assert captured["train/avoidable_joker_melds_mean"] == 0.5


def test_parse_curriculum_ratios_normalizes_values():
    r1, r2, r3 = _parse_curriculum_ratios("2,3,5")
    assert r1 == pytest.approx(0.2)
    assert r2 == pytest.approx(0.3)
    assert r3 == pytest.approx(0.5)


def test_parse_curriculum_ratios_rejects_invalid_input():
    with pytest.raises(ValueError):
        _parse_curriculum_ratios("0.2,0.8")
    with pytest.raises(ValueError):
        _parse_curriculum_ratios("0.2,0,-0.1")

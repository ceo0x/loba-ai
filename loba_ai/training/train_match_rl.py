from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback

from loba_ai.match_env import MatchLobaEnv
from loba_ai.model_io import load_model
from loba_ai.rules import Rules


DEFAULT_OPPONENT_MODEL_PATH = "artifacts/loba_match_maskable_ppo.zip"


class MatchRoundsLoggingCallback(BaseCallback):
    def __init__(self, window_size: int = 100) -> None:
        super().__init__()
        self.window_size = max(1, int(window_size))
        self._rounds_buffer: deque[float] = deque(maxlen=self.window_size)
        self._joker_meld_rate_buffer: deque[float] = deque(maxlen=self.window_size)
        self._joker_hold_rate_buffer: deque[float] = deque(maxlen=self.window_size)
        self._joker_discards_buffer: deque[float] = deque(maxlen=self.window_size)
        self._avoidable_joker_melds_buffer: deque[float] = deque(maxlen=self.window_size)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if info.get("match_finished") and "rounds_played" in info:
                self._rounds_buffer.append(float(info["rounds_played"]))
            if info.get("match_finished"):
                self._joker_meld_rate_buffer.append(float(info.get("joker_meld_rate", 0.0)))
                self._joker_hold_rate_buffer.append(float(info.get("joker_hold_rate", 0.0)))
                self._joker_discards_buffer.append(float(info.get("joker_discards", 0.0)))
                self._avoidable_joker_melds_buffer.append(float(info.get("avoidable_joker_melds", 0.0)))
        return True

    def _on_rollout_end(self) -> None:
        if not self._rounds_buffer:
            return
        rounds_mean = sum(self._rounds_buffer) / float(len(self._rounds_buffer))
        self.logger.record("train/ep_rounds_mean", rounds_mean)
        if self._joker_meld_rate_buffer:
            self.logger.record(
                "train/joker_meld_rate_mean",
                sum(self._joker_meld_rate_buffer) / float(len(self._joker_meld_rate_buffer)),
            )
        if self._joker_hold_rate_buffer:
            self.logger.record(
                "train/joker_hold_rate_mean",
                sum(self._joker_hold_rate_buffer) / float(len(self._joker_hold_rate_buffer)),
            )
        if self._joker_discards_buffer:
            self.logger.record(
                "train/joker_discards_mean",
                sum(self._joker_discards_buffer) / float(len(self._joker_discards_buffer)),
            )
        if self._avoidable_joker_melds_buffer:
            self.logger.record(
                "train/avoidable_joker_melds_mean",
                sum(self._avoidable_joker_melds_buffer) / float(len(self._avoidable_joker_melds_buffer)),
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL model for Loba match play (to target points)")
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--model-out", type=str, default="artifacts/loba_match_maskable_ppo")
    parser.add_argument("--opponent", type=str, choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--table-mode", type=str, choices=["classic", "mixed4p"], default="classic")
    parser.add_argument("--opponent-model-path", type=str, default=None)
    parser.add_argument("--default-opponent-model-path", type=str, default=DEFAULT_OPPONENT_MODEL_PATH)
    parser.add_argument("--target-points", type=int, default=100)
    parser.add_argument("--max-rounds-per-match", type=int, default=64)
    parser.add_argument("--round-score-delta-coef", type=float, default=5.0)
    parser.add_argument("--match-win-bonus", type=float, default=150.0)
    parser.add_argument("--match-loss-penalty", type=float, default=150.0)
    parser.add_argument("--meld-joker-penalty", type=float, default=0.35)
    parser.add_argument("--avoidable-joker-extra-penalty", type=float, default=0.45)
    parser.add_argument("--joker-discard-penalty", type=float, default=0.75)
    parser.add_argument("--joker-hold-bonus", type=float, default=0.05)
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Train in stages: random -> heuristic -> mixed4p (if opponent model available).",
    )
    parser.add_argument(
        "--curriculum-ratios",
        type=str,
        default="0.2,0.5,0.3",
        help="Comma-separated stage ratios for curriculum training.",
    )
    return parser.parse_args()


def resolve_opponent_model_path(args: argparse.Namespace) -> str | None:
    if args.table_mode != "mixed4p":
        return None
    opponent_model_path = args.opponent_model_path or args.default_opponent_model_path
    if not opponent_model_path:
        raise ValueError("mixed4p requires --opponent-model-path or --default-opponent-model-path")
    return opponent_model_path


def _parse_curriculum_ratios(raw: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("--curriculum-ratios must contain exactly 3 values, e.g. 0.2,0.5,0.3")
    values = tuple(float(p) for p in parts)
    if any(v <= 0.0 for v in values):
        raise ValueError("--curriculum-ratios values must be > 0")
    total = sum(values)
    return values[0] / total, values[1] / total, values[2] / total


def _build_env(args: argparse.Namespace, opponent: str, table_mode: str, trained_opponent_model):
    rules = Rules(num_players=4) if table_mode == "mixed4p" else Rules()
    return MatchLobaEnv(
        rules=rules,
        opponent=opponent,
        table_mode=table_mode,
        trained_opponent_model=trained_opponent_model,
        target_points=args.target_points,
        max_rounds_per_match=args.max_rounds_per_match,
        round_score_delta_coef=args.round_score_delta_coef,
        match_win_bonus=args.match_win_bonus,
        match_loss_penalty=args.match_loss_penalty,
        meld_joker_penalty=args.meld_joker_penalty,
        avoidable_joker_extra_penalty=args.avoidable_joker_extra_penalty,
        joker_discard_penalty=args.joker_discard_penalty,
        joker_hold_bonus=args.joker_hold_bonus,
    )


def main() -> None:
    args = parse_args()
    trained_opponent_model = None
    if args.table_mode == "mixed4p":
        opponent_model_path = resolve_opponent_model_path(args)
        assert opponent_model_path is not None
        if not Path(opponent_model_path).exists():
            raise FileNotFoundError(f"Opponent model not found: {opponent_model_path}")
        trained_opponent_model = load_model(opponent_model_path)
    elif args.curriculum:
        candidate_path = args.opponent_model_path or args.default_opponent_model_path
        if candidate_path and Path(candidate_path).exists():
            trained_opponent_model = load_model(candidate_path)
    env = _build_env(
        args=args,
        opponent=args.opponent,
        table_mode=args.table_mode,
        trained_opponent_model=trained_opponent_model,
    )

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
    )

    callback = MatchRoundsLoggingCallback()
    if args.curriculum:
        r_random, r_heuristic, r_mixed = _parse_curriculum_ratios(args.curriculum_ratios)
        stages = [
            ("random", "classic", r_random),
            ("heuristic", "classic", r_heuristic),
        ]
        if trained_opponent_model is not None:
            stages.append(("heuristic", "mixed4p", r_mixed))
        else:
            stages.append(("heuristic", "classic", r_mixed))
        learned_steps = 0
        for stage_ix, (opponent, table_mode, ratio) in enumerate(stages):
            stage_steps = max(1, int(args.timesteps * ratio))
            if stage_ix == len(stages) - 1:
                stage_steps = max(1, args.timesteps - learned_steps)
            learned_steps += stage_steps
            stage_env = _build_env(
                args=args,
                opponent=opponent,
                table_mode=table_mode,
                trained_opponent_model=trained_opponent_model,
            )
            model.set_env(stage_env)
            model.learn(
                total_timesteps=stage_steps,
                progress_bar=True,
                callback=callback,
                reset_num_timesteps=False,
            )
    else:
        model.learn(total_timesteps=args.timesteps, progress_bar=True, callback=callback)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)

    obs, info = env.reset()
    _ = model.predict(obs, deterministic=True, action_masks=get_action_masks(env))
    print(f"Model saved to {args.model_out}.zip")


if __name__ == "__main__":
    main()

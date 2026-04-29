from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks

from loba_ai.match_env import MatchLobaEnv
from loba_ai.model_io import choose_action, load_model
from loba_ai.rules import Rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and rank match checkpoints with fixed seeds.")
    parser.add_argument("--models", nargs="+", required=True, help="Model checkpoint paths (.zip or stem).")
    parser.add_argument("--episodes", type=int, default=120, help="Episodes per model.")
    parser.add_argument("--seeds", type=str, default="11,23,47,71,97", help="Comma-separated evaluation seeds.")
    parser.add_argument("--opponent", type=str, choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--table-mode", type=str, choices=["classic", "mixed4p"], default="classic")
    parser.add_argument("--target-points", type=int, default=100)
    parser.add_argument("--max-rounds-per-match", type=int, default=64)
    parser.add_argument("--json-out", type=str, default="", help="Optional path to save JSON report.")
    return parser.parse_args()


def _normalize_model_path(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    zipped = p.with_suffix(".zip")
    if zipped.exists():
        return str(zipped)
    raise FileNotFoundError(f"Model not found: {path}")


def _composite_score(metrics: dict[str, float]) -> float:
    # Higher is better.
    return (
        (metrics["win_rate"] * 100.0)
        - metrics["avg_match_score"]
        + (metrics["avg_joker_hold_rate"] * 10.0)
        - (metrics["avg_joker_meld_rate"] * 15.0)
        - (metrics["avg_joker_discards"] * 2.0)
        - (metrics["avg_avoidable_joker_melds"] * 4.0)
    )


def evaluate_model(model_path: str, args: argparse.Namespace, seeds: list[int]) -> dict:
    model = load_model(model_path)
    rules = Rules(num_players=4) if args.table_mode == "mixed4p" else Rules()
    env = MatchLobaEnv(
        rules=rules,
        opponent=args.opponent,
        table_mode=args.table_mode,
        trained_opponent_model=model if args.table_mode == "mixed4p" else None,
        target_points=args.target_points,
        max_rounds_per_match=args.max_rounds_per_match,
    )

    wins = 0
    match_scores: list[float] = []
    joker_hold_rates: list[float] = []
    joker_meld_rates: list[float] = []
    joker_discards: list[float] = []
    avoidable_joker_melds: list[float] = []

    for episode_ix in range(args.episodes):
        seed = seeds[episode_ix % len(seeds)]
        obs, _ = env.reset(seed=seed)
        done = False
        final_info: dict = {}
        while not done:
            action = choose_action(model, obs, get_action_masks(env).astype(bool))
            obs, _, done, _, info = env.step(action)
            final_info = info
        if int(final_info.get("match_winner", -1)) == 0:
            wins += 1
        scores = final_info.get("match_scores", [0.0])
        match_scores.append(float(scores[0]))
        joker_hold_rates.append(float(final_info.get("joker_hold_rate", 0.0)))
        joker_meld_rates.append(float(final_info.get("joker_meld_rate", 0.0)))
        joker_discards.append(float(final_info.get("joker_discards", 0.0)))
        avoidable_joker_melds.append(float(final_info.get("avoidable_joker_melds", 0.0)))

    metrics = {
        "model_path": model_path,
        "episodes": int(args.episodes),
        "win_rate": wins / float(args.episodes),
        "avg_match_score": float(np.mean(match_scores)),
        "avg_joker_hold_rate": float(np.mean(joker_hold_rates)),
        "avg_joker_meld_rate": float(np.mean(joker_meld_rates)),
        "avg_joker_discards": float(np.mean(joker_discards)),
        "avg_avoidable_joker_melds": float(np.mean(avoidable_joker_melds)),
    }
    metrics["composite_score"] = _composite_score(metrics)
    return metrics


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("At least one evaluation seed is required.")

    evaluated = [evaluate_model(_normalize_model_path(p), args, seeds) for p in args.models]
    ranked = sorted(evaluated, key=lambda x: x["composite_score"], reverse=True)

    print("=== Checkpoint ranking ===")
    for idx, row in enumerate(ranked, start=1):
        print(
            f"{idx:>2}. {row['model_path']} | score={row['composite_score']:.3f} "
            f"| win_rate={row['win_rate']:.3f} | avg_match_score={row['avg_match_score']:.2f} "
            f"| joker_meld_rate={row['avg_joker_meld_rate']:.3f} | joker_hold_rate={row['avg_joker_hold_rate']:.3f}"
        )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"ranking": ranked}, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {out_path}")


if __name__ == "__main__":
    main()

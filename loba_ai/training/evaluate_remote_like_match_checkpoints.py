from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks

from loba_ai.model_io import load_model
from loba_ai.remote_like_match_smart_env import RemoteLikeMatchSmartLobaEnv
from loba_ai.rules import Rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate remote-like match-aware checkpoints with fixed seeds."
    )
    parser.add_argument("--models", nargs="+", required=True, help="Model checkpoint paths.")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes per model.")
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Optional comma-separated seeds. Defaults to --seed + episode index.",
    )
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-players", type=int, choices=[2, 3, 4, 5], default=3)
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["random", "heuristic", "strong_heuristic", "mixed_heuristic"],
        default="mixed_heuristic",
    )
    parser.add_argument("--discard-history-window", type=int, default=2)
    parser.add_argument("--target-points", type=int, default=100)
    parser.add_argument("--max-rounds-per-match", type=int, default=64)
    parser.add_argument("--max-reenganches", type=int, default=2)
    parser.add_argument(
        "--no-action-tactical-features",
        action="store_true",
        help="Use the old 799-dim observation layout for legacy checkpoints.",
    )
    parser.add_argument("--max-steps-per-episode", type=int, default=5000)
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def _normalize_model_path(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    zipped = p.with_suffix(".zip")
    if zipped.exists():
        return str(zipped)
    raise FileNotFoundError(f"Model not found: {path}")


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds.strip():
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if not seeds:
            raise ValueError("At least one seed is required when --seeds is provided.")
        return seeds
    return [int(args.seed) + i for i in range(int(args.episodes))]


def _rank_agent(match_scores: list[float]) -> int:
    agent_score = float(match_scores[0])
    return sorted(float(s) for s in match_scores).index(agent_score) + 1


def evaluate_model(model_path: str, args: argparse.Namespace, seeds: list[int]) -> dict[str, Any]:
    print(f"[eval] starting {model_path} ({args.episodes} episodes)", flush=True)
    model = load_model(model_path)
    env = RemoteLikeMatchSmartLobaEnv(
        rules=Rules(num_players=int(args.num_players)),
        seed=int(args.seed),
        opponent=args.opponent,
        discard_history_window=int(args.discard_history_window),
        target_points=int(args.target_points),
        max_rounds_per_match=int(args.max_rounds_per_match),
        max_reenganches=int(args.max_reenganches),
        enable_action_tactical_features=not bool(args.no_action_tactical_features),
    )

    wins: list[int] = []
    rewards: list[float] = []
    rounds: list[int] = []
    eliminated: list[int] = []
    elimination_rounds: list[int] = []
    agent_scores: list[float] = []
    ranks: list[int] = []
    n_active: list[int] = []
    timeouts = 0
    missing_winner = 0

    for episode_ix in range(int(args.episodes)):
        if episode_ix > 0 and episode_ix % 25 == 0:
            print(
                f"[eval] {Path(model_path).name}: {episode_ix}/{args.episodes} episodes",
                flush=True,
            )
        seed = seeds[episode_ix % len(seeds)]
        obs, _ = env.reset(seed=seed)
        done = False
        ep_reward = 0.0
        final_info: dict[str, Any] = {}
        steps = 0

        while not done and steps < int(args.max_steps_per_episode):
            masks = get_action_masks(env)
            action, _ = model.predict(obs, deterministic=True, action_masks=masks)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_reward += float(reward)
            done = bool(terminated or truncated)
            final_info = info
            steps += 1

        if not done:
            timeouts += 1
        winner = final_info.get("match_winner")
        if winner is None:
            missing_winner += 1
        match_scores = final_info.get("match_scores") or [0.0] * int(args.num_players)
        agent_eliminated = bool(final_info.get("agent_eliminated", False))
        elim_round = final_info.get("agent_elimination_round")

        wins.append(1 if winner == 0 else 0)
        rewards.append(ep_reward)
        rounds.append(int(final_info.get("rounds_played", 0)))
        eliminated.append(1 if agent_eliminated else 0)
        if agent_eliminated and isinstance(elim_round, int):
            elimination_rounds.append(elim_round)
        agent_scores.append(float(match_scores[0]))
        ranks.append(_rank_agent([float(s) for s in match_scores]))
        n_active.append(int(final_info.get("n_active_players", 0)))

    metrics = {
        "model_path": model_path,
        "episodes": int(args.episodes),
        "opponent": args.opponent,
        "win_rate": float(np.mean(wins)),
        "mean_reward": float(np.mean(rewards)),
        "avg_match_score": float(np.mean(agent_scores)),
        "agent_elimination_rate": float(np.mean(eliminated)),
        "agent_avg_elimination_round": float(np.mean(elimination_rounds)) if elimination_rounds else None,
        "avg_rounds": float(np.mean(rounds)),
        "avg_rank": float(np.mean(ranks)),
        "rank_1_rate": float(np.mean([1 if r == 1 else 0 for r in ranks])),
        "rank_3plus_rate": float(np.mean([1 if r >= 3 else 0 for r in ranks])),
        "avg_active_players_at_end": float(np.mean(n_active)),
        "timeout_rate": timeouts / float(args.episodes),
        "missing_winner_rate": missing_winner / float(args.episodes),
    }
    print(
        f"[eval] done {Path(model_path).name}: "
        f"win_rate={metrics['win_rate']:.3f} "
        f"mean_reward={metrics['mean_reward']:.1f} "
        f"elim={metrics['agent_elimination_rate']:.3f} "
        f"timeouts={metrics['timeout_rate']:.3f}",
        flush=True,
    )
    return metrics


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args)
    rows = [evaluate_model(_normalize_model_path(p), args, seeds) for p in args.models]
    ranked = sorted(
        rows,
        key=lambda x: (
            x["win_rate"],
            x["mean_reward"],
            -x["agent_elimination_rate"],
            -x["avg_match_score"],
        ),
        reverse=True,
    )

    print("=== Remote-like match checkpoint ranking ===")
    for idx, row in enumerate(ranked, start=1):
        elim_round = row["agent_avg_elimination_round"]
        elim_round_s = "n/a" if elim_round is None else f"{elim_round:.2f}"
        print(
            f"{idx:>2}. {row['model_path']} | "
            f"win_rate={row['win_rate']:.3f} | "
            f"mean_reward={row['mean_reward']:.1f} | "
            f"avg_score={row['avg_match_score']:.2f} | "
            f"elim={row['agent_elimination_rate']:.3f} | "
            f"elim_round={elim_round_s} | "
            f"avg_rank={row['avg_rank']:.2f}"
        )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"ranking": ranked}, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {out_path}")


if __name__ == "__main__":
    main()

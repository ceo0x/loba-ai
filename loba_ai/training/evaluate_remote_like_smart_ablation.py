from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib.common.maskable.utils import get_action_masks

from loba_ai.model_io import choose_action, load_model
from loba_ai.remote_like_smart_env import RemoteLikeSmartLobaEnv
from loba_ai.rules import Rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate remote-like smart ablation checkpoints.")
    parser.add_argument("--models", nargs="+", required=True, help="Model paths (.zip or stem).")
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--num-players", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="11,23,47,71,97")
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


def _parse_profile(path: str) -> str:
    name = Path(path).stem.lower()
    for key in ("terminal_only", "light", "full"):
        if key in name:
            return key
    return "unknown"


def evaluate_model(model_path: str, episodes: int, seeds: list[int], num_players: int) -> dict[str, Any]:
    model = load_model(model_path)
    env = RemoteLikeSmartLobaEnv(rules=Rules(num_players=num_players), seed=0, opponent="random")

    wins = 0
    final_hand_points: list[float] = []
    discards_when_play: list[float] = []
    plays_when_available: list[float] = []
    per_episode_rewards: list[float] = []

    for ep_ix in range(episodes):
        seed = seeds[ep_ix % len(seeds)]
        obs, _ = env.reset(seed=seed)
        done = False
        final_info: dict[str, Any] = {}
        reward_acc = 0.0
        while not done:
            action = choose_action(model, obs, get_action_masks(env).astype(bool))
            obs, reward, done, _, info = env.step(action)
            reward_acc += float(reward)
            final_info = info
        per_episode_rewards.append(reward_acc)
        if int(env.engine.state.winner if env.engine.state.winner is not None else -1) == 0:
            wins += 1
        final_hand_points.append(float(final_info.get("hand_points", 0.0)))
        disc = float(final_info.get("episode_discard_with_play_options", 0.0))
        turns = float(final_info.get("episode_play_options_turns", 0.0))
        play = float(final_info.get("episode_play_when_available", 0.0))
        discards_when_play.append((disc / turns) if turns > 0 else 0.0)
        plays_when_available.append((play / turns) if turns > 0 else 0.0)

    return {
        "model_path": model_path,
        "profile": _parse_profile(model_path),
        "episodes": int(episodes),
        "win_rate": wins / float(episodes),
        "avg_final_hand_points": float(np.mean(final_hand_points)),
        "avg_discard_when_play_ratio": float(np.mean(discards_when_play)),
        "avg_play_when_available_ratio": float(np.mean(plays_when_available)),
        "avg_episode_reward": float(np.mean(per_episode_rewards)),
        "reward_std": float(np.std(per_episode_rewards)),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["profile"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for profile, vals in grouped.items():
        summary.append(
            {
                "profile": profile,
                "models": len(vals),
                "win_rate_mean": float(np.mean([v["win_rate"] for v in vals])),
                "win_rate_std": float(np.std([v["win_rate"] for v in vals])),
                "avg_final_hand_points_mean": float(np.mean([v["avg_final_hand_points"] for v in vals])),
                "avg_play_when_available_ratio_mean": float(np.mean([v["avg_play_when_available_ratio"] for v in vals])),
                "avg_discard_when_play_ratio_mean": float(np.mean([v["avg_discard_when_play_ratio"] for v in vals])),
                "avg_episode_reward_mean": float(np.mean([v["avg_episode_reward"] for v in vals])),
            }
        )
    summary.sort(key=lambda x: x["win_rate_mean"], reverse=True)
    return summary


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("At least one evaluation seed is required.")
    model_paths = [_normalize_model_path(p) for p in args.models]
    rows = [evaluate_model(p, args.episodes, seeds, args.num_players) for p in model_paths]
    summary = summarize(rows)

    print("=== Remote-like smart ablation results (per model) ===")
    for row in rows:
        print(
            f"{row['profile']:>13} | {row['model_path']} | "
            f"win_rate={row['win_rate']:.3f} | hand_pts={row['avg_final_hand_points']:.2f} | "
            f"play_ratio={row['avg_play_when_available_ratio']:.3f} | discard_ratio={row['avg_discard_when_play_ratio']:.3f}"
        )
    print("=== Aggregated by profile ===")
    for row in summary:
        print(
            f"{row['profile']:>13} | win={row['win_rate_mean']:.3f} (+/-{row['win_rate_std']:.3f}) | "
            f"hand_pts={row['avg_final_hand_points_mean']:.2f} | play_ratio={row['avg_play_when_available_ratio_mean']:.3f}"
        )

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"models": rows, "summary": summary}, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {out}")


if __name__ == "__main__":
    main()

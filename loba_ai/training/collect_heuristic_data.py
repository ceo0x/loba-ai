"""Collect (obs, action_mask, action_idx) tuples from the heuristic agent for BC.

The heuristic plays as seat 0 against opponents (random/heuristic/mixed) so we capture
its decisions in the same observation space the RL model will see.

Output: a single .npz with arrays:
    obs:     (N, obs_dim) float32
    masks:   (N, action_space) int8
    actions: (N,)           int64

Usage:
    .venv/bin/python -m loba_ai.training.collect_heuristic_data \\
        --num-players 3 --opponent heuristic \\
        --target-samples 200000 \\
        --output artifacts/bc_dataset_heuristic_3p.npz \\
        --seed 1
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from loba_ai.agents.remote_like_heuristic_agent import (
    RemoteLikeHeuristicAgent,
    StrongRemoteLikeHeuristicAgent,
)
from loba_ai.remote_like_match_smart_env import RemoteLikeMatchSmartLobaEnv
from loba_ai.rules import Rules


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect BC dataset from heuristic agent in seat 0")
    p.add_argument("--num-players", type=int, choices=[2, 3, 4, 5], default=3)
    p.add_argument(
        "--opponent",
        type=str,
        choices=["random", "heuristic", "strong_heuristic", "mixed_heuristic"],
        default="heuristic",
        help="Opponent type the heuristic plays against (NOT what plays seat 0). Ignored when --opponent-mix is set.",
    )
    p.add_argument(
        "--opponent-mix",
        type=str,
        default=None,
        help=(
            "Comma-separated list of opponent types (e.g. 'random,heuristic,strong_heuristic,mixed_heuristic'). "
            "When set, each match samples a fresh opponent uniformly from this list, broadening "
            "the state distribution covered by the dataset (mitigates BC covariate shift)."
        ),
    )
    p.add_argument(
        "--skip-trivial",
        action="store_true",
        help=(
            "Skip recording samples where len(legal_actions)==1. Those decisions carry no "
            "learning signal under masking (gradient = 0 with one legal action) and just "
            "inflate the dataset. With this flag, every sample stored is a real choice."
        ),
    )
    p.add_argument("--max-reenganches", type=int, default=2)
    p.add_argument("--target-points", type=int, default=100)
    p.add_argument("--discard-history-window", type=int, default=2)
    p.add_argument(
        "--seat-0-policy",
        type=str,
        choices=["heuristic", "strong_heuristic"],
        default="heuristic",
        help="Which heuristic plays seat 0 — its decisions become the dataset.",
    )
    p.add_argument(
        "--target-samples",
        type=int,
        default=200_000,
        help="Approximate number of (obs, action) decisions to record.",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output .npz path. Parent directory will be created.",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--report-every",
        type=int,
        default=10_000,
        help="Print progress every N samples collected.",
    )
    return p.parse_args()


def _parse_opponent_mix(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    valid = {"random", "heuristic", "strong_heuristic", "mixed_heuristic"}
    pool = [s.strip() for s in raw.split(",") if s.strip()]
    bad = [p for p in pool if p not in valid]
    if bad:
        raise ValueError(f"--opponent-mix contains invalid opponent(s): {bad}. Valid: {sorted(valid)}")
    if not pool:
        raise ValueError("--opponent-mix is empty after parsing.")
    return pool


def _build_env(args: argparse.Namespace, opponent: str, seed: int) -> RemoteLikeMatchSmartLobaEnv:
    return RemoteLikeMatchSmartLobaEnv(
        rules=Rules(num_players=int(args.num_players)),
        seed=int(seed),
        opponent=opponent,
        discard_history_window=int(args.discard_history_window),
        target_points=int(args.target_points),
        max_reenganches=int(args.max_reenganches),
    )


def main() -> None:
    args = parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    opponent_mix = _parse_opponent_mix(args.opponent_mix)
    rng = np.random.default_rng(int(args.seed))

    def _sample_opponent() -> str:
        if opponent_mix is None:
            return args.opponent
        return str(rng.choice(opponent_mix))

    current_opponent = _sample_opponent()
    env = _build_env(args, current_opponent, seed=int(args.seed))

    if args.seat_0_policy == "strong_heuristic":
        seat0 = StrongRemoteLikeHeuristicAgent(rng=rng)
    else:
        seat0 = RemoteLikeHeuristicAgent(rng=rng)

    obs_dim = int(env.observation_space.shape[0])
    action_space_n = int(env.action_space.n)
    target = int(args.target_samples)

    obs_buf = np.zeros((target, obs_dim), dtype=np.float32)
    mask_buf = np.zeros((target, action_space_n), dtype=np.int8)
    action_buf = np.zeros((target,), dtype=np.int64)

    n_collected = 0
    n_matches = 0
    n_seen_decisions = 0  # total decisions (including skipped trivials) — for rate reporting
    n_skipped_trivial = 0
    opponent_match_counts: dict[str, int] = {}
    t_start = time.time()
    last_report_n = 0

    obs, info = env.reset(seed=int(args.seed))
    opponent_match_counts[current_opponent] = opponent_match_counts.get(current_opponent, 0) + 1

    while n_collected < target:
        legal = list(env._last_legal_actions)
        mask = env._action_mask()  # (256,) int8

        if not legal:
            # Defensive — shouldn't happen mid-match in seat 0; reset.
            obs, info = env.reset()
            continue

        # Heuristic picks an index over `legal`.
        hand_payload = [env._card_payload(c) for c in env.engine.state.players[0].hand]
        phase = str(env.engine.state.phase)
        try:
            idx = int(seat0.act(legal, hand_payload, phase))
        except Exception:
            idx = 0
        idx = max(0, min(idx, len(legal) - 1))

        n_seen_decisions += 1
        is_trivial = len(legal) == 1
        if args.skip_trivial and is_trivial:
            # Execute the forced action to advance the match, but don't record.
            n_skipped_trivial += 1
        else:
            # Record the (obs, mask, idx) BEFORE the env steps.
            obs_buf[n_collected] = obs
            mask_buf[n_collected] = mask
            action_buf[n_collected] = idx
            n_collected += 1

        obs, reward, terminated, truncated, info = env.step(idx)
        if terminated or truncated:
            n_matches += 1
            # Resample opponent for the next match (no-op when opponent_mix is None).
            new_opponent = _sample_opponent()
            if new_opponent != current_opponent:
                env = _build_env(args, new_opponent, seed=int(args.seed) + n_matches)
                current_opponent = new_opponent
            obs, info = env.reset()
            opponent_match_counts[current_opponent] = opponent_match_counts.get(current_opponent, 0) + 1

        if n_collected - last_report_n >= int(args.report_every):
            elapsed = time.time() - t_start
            rate = n_collected / max(1e-9, elapsed)
            eta = (target - n_collected) / max(1e-9, rate)
            extra = ""
            if args.skip_trivial and n_seen_decisions:
                skip_pct = 100.0 * n_skipped_trivial / n_seen_decisions
                extra = f"  skipped_trivial={n_skipped_trivial} ({skip_pct:.1f}%)"
            print(
                f"[collect] n={n_collected:>8d}/{target}  matches={n_matches}  "
                f"rate={rate:.0f}/s  elapsed={elapsed:.1f}s  eta={eta:.0f}s{extra}"
            )
            last_report_n = n_collected

    elapsed = time.time() - t_start
    print(f"[collect] done. n={n_collected}  matches={n_matches}  elapsed={elapsed:.1f}s")
    if args.skip_trivial:
        print(
            f"[collect] skip-trivial: dropped {n_skipped_trivial}/{n_seen_decisions} "
            f"({100.0*n_skipped_trivial/max(1,n_seen_decisions):.1f}%) trivial decisions."
        )
    if opponent_mix is not None:
        print(f"[collect] opponent-mix breakdown: {opponent_match_counts}")

    print(f"[collect] saving {out_path} ...")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp.npz")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
        np.savez_compressed(
            tmp_path,
            obs=obs_buf,
            masks=mask_buf,
            actions=action_buf,
            meta=np.array(
                {
                    "obs_dim": obs_dim,
                    "action_space_n": action_space_n,
                    "num_players": int(args.num_players),
                    "opponent": args.opponent,
                    "opponent_mix": opponent_mix,
                    "opponent_match_counts": opponent_match_counts,
                    "seat_0_policy": args.seat_0_policy,
                    "skip_trivial": bool(args.skip_trivial),
                    "n_skipped_trivial": int(n_skipped_trivial),
                    "n_seen_decisions": int(n_seen_decisions),
                    "target_points": int(args.target_points),
                    "max_reenganches": int(args.max_reenganches),
                    "discard_history_window": int(args.discard_history_window),
                    "seed": int(args.seed),
                    "n_samples": int(n_collected),
                    "n_matches": int(n_matches),
                },
                dtype=object,
            ),
        )
        os.replace(tmp_path, out_path)
    except BaseException:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise
    print(f"[collect] saved {out_path}  size={out_path.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()

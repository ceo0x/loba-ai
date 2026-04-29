from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run remote-like smart shaping ablation training batches.")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--num-players", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="11,23")
    parser.add_argument("--artifact-dir", type=str, default="artifacts/ablation_remote_like_smart")
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--vec-env", type=str, choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_seeds(raw: str) -> list[int]:
    out = [int(s.strip()) for s in raw.split(",") if s.strip()]
    if not out:
        raise ValueError("No seeds provided.")
    return out


def _train_cmd(
    python_bin: str,
    timesteps: int,
    num_players: int,
    model_out: str,
    seed: int,
    profile: str,
    n_envs: int,
    vec_env: str,
    n_steps: int,
    batch_size: int,
) -> list[str]:
    cmd = [
        python_bin,
        "-m",
        "loba_ai.training.train_remote_like_smart_rl",
        "--timesteps",
        str(timesteps),
        "--num-players",
        str(num_players),
        "--model-out",
        model_out,
        "--n-envs",
        str(max(1, int(n_envs))),
        "--vec-env",
        vec_env,
        "--n-steps",
        str(max(64, int(n_steps))),
        "--batch-size",
        str(max(64, int(batch_size))),
    ]

    if profile == "light":
        cmd.extend(
            [
                "--reward-discard-with-play-penalty",
                "0.4",
                "--reward-discard-project-penalty",
                "0.2",
                "--reward-discard-low-before-high-penalty",
                "0.1",
                "--reward-discard-hot-card-penalty",
                "0.1",
                "--reward-play-bonus",
                "0.3",
                "--reward-extend-bonus",
                "0.2",
                "--reward-cruzar-bonus",
                "0.15",
            ]
        )
    elif profile == "terminal_only":
        cmd.extend(
            [
                "--reward-discard-with-play-penalty",
                "0.0",
                "--reward-discard-project-penalty",
                "0.0",
                "--reward-discard-low-before-high-penalty",
                "0.0",
                "--reward-discard-hot-card-penalty",
                "0.0",
                "--reward-play-bonus",
                "0.0",
                "--reward-extend-bonus",
                "0.0",
                "--reward-cruzar-bonus",
                "0.0",
            ]
        )

    cmd.extend(["--seed", str(seed)])
    return cmd


def main() -> None:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    profiles = ("full", "light", "terminal_only")
    commands: list[list[str]] = []
    for seed in seeds:
        for profile in profiles:
            model_out = str(artifact_dir / f"remote_like_smart_{profile}_p{args.num_players}_seed{seed}")
            commands.append(
                _train_cmd(
                    args.python_bin,
                    args.timesteps,
                    args.num_players,
                    model_out,
                    seed,
                    profile,
                    args.n_envs,
                    args.vec_env,
                    args.n_steps,
                    args.batch_size,
                )
            )

    for cmd in commands:
        print("$ " + " ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

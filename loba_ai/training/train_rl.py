from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

from loba_ai.env import LobaEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL model for Loba")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--model-out", type=str, default="artifacts/loba_maskable_ppo")
    parser.add_argument("--opponent", type=str, choices=["heuristic", "random"], default="heuristic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = LobaEnv(opponent=args.opponent)

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

    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)

    obs, info = env.reset()
    _ = model.predict(obs, deterministic=True, action_masks=get_action_masks(env))
    print(f"Model saved to {args.model_out}.zip")


if __name__ == "__main__":
    main()

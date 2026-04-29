from __future__ import annotations

import argparse

import numpy as np

from loba_ai.env import LobaEnv


def run_episode(env: LobaEnv, rng: np.random.Generator) -> float:
    obs, info = env.reset()
    done = False
    total = 0.0
    while not done:
        valid = np.flatnonzero(info["action_mask"])
        action = int(rng.choice(valid))
        obs, reward, done, _, info = env.step(action)
        total += reward
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    env = LobaEnv(opponent="heuristic")
    rng = np.random.default_rng(123)
    rewards = [run_episode(env, rng) for _ in range(args.episodes)]
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {np.mean(rewards):.3f}")
    print(f"Win-ish episodes (reward > 0): {sum(r > 0 for r in rewards)}")


if __name__ == "__main__":
    main()

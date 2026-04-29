from __future__ import annotations

import argparse
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from loba_ai.remote_like_smart_env import RemoteLikeSmartLobaEnv
from loba_ai.rules import Rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train smart RL model for remote-like Loba protocol")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--model-out", type=str, default="artifacts/loba_remote_like_smart_maskable_ppo")
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--opponent", type=str, choices=["random"], default="random")
    parser.add_argument("--discard-history-window", type=int, default=2)
    parser.add_argument("--reward-play-bonus", type=float, default=0.7)
    parser.add_argument("--reward-extend-bonus", type=float, default=0.5)
    parser.add_argument("--reward-cruzar-bonus", type=float, default=0.4)
    parser.add_argument("--reward-discard-with-play-penalty", type=float, default=1.1)
    parser.add_argument("--reward-discard-project-penalty", type=float, default=0.5)
    parser.add_argument("--reward-discard-low-before-high-penalty", type=float, default=0.2)
    parser.add_argument("--reward-discard-hot-card-penalty", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments.")
    parser.add_argument("--vec-env", type=str, choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def _build_env(args: argparse.Namespace, rules: Rules):
    n_envs = max(1, int(args.n_envs))

    def _make_env(rank: int):
        def _init():
            env = RemoteLikeSmartLobaEnv(
                rules=rules,
                seed=int(args.seed) + rank,
                opponent=args.opponent,
                discard_history_window=max(1, int(args.discard_history_window)),
            )
            env.reward_play_action_bonus = float(args.reward_play_bonus)
            env.reward_extend_action_bonus = float(args.reward_extend_bonus)
            env.reward_cruzar_action_bonus = float(args.reward_cruzar_bonus)
            env.reward_discard_with_play_options_penalty = float(args.reward_discard_with_play_penalty)
            env.reward_discard_project_penalty = float(args.reward_discard_project_penalty)
            env.reward_discard_low_single_before_high_single_penalty = float(args.reward_discard_low_before_high_penalty)
            env.reward_discard_hot_card_penalty = float(args.reward_discard_hot_card_penalty)
            return env

        return _init

    env_fns = [_make_env(i) for i in range(n_envs)]
    if n_envs == 1 or args.vec_env == "dummy":
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method="spawn")


def main() -> None:
    args = parse_args()
    rules = Rules(num_players=max(2, min(5, int(args.num_players))))
    env = _build_env(args, rules)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=max(64, int(args.n_steps)),
        batch_size=max(64, int(args.batch_size)),
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        seed=int(args.seed),
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=True)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    env.close()

    check_env = RemoteLikeSmartLobaEnv(
        rules=rules,
        seed=int(args.seed),
        opponent=args.opponent,
        discard_history_window=max(1, int(args.discard_history_window)),
    )
    check_env.reward_play_action_bonus = float(args.reward_play_bonus)
    check_env.reward_extend_action_bonus = float(args.reward_extend_bonus)
    check_env.reward_cruzar_action_bonus = float(args.reward_cruzar_bonus)
    check_env.reward_discard_with_play_options_penalty = float(args.reward_discard_with_play_penalty)
    check_env.reward_discard_project_penalty = float(args.reward_discard_project_penalty)
    check_env.reward_discard_low_single_before_high_single_penalty = float(args.reward_discard_low_before_high_penalty)
    check_env.reward_discard_hot_card_penalty = float(args.reward_discard_hot_card_penalty)
    obs, _ = check_env.reset()
    _ = model.predict(obs, deterministic=True, action_masks=get_action_masks(check_env))
    check_env.close()
    print(f"Remote-like smart model saved to {args.model_out}.zip")


if __name__ == "__main__":
    main()

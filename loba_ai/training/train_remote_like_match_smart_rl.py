from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from loba_ai.model_io import load_model
from loba_ai.remote_like_match_smart_env import RemoteLikeMatchSmartLobaEnv
from loba_ai.rules import Rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train smart RL model for match-aware Loba (remote-like protocol)")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--model-out", type=str, default="artifacts/loba_remote_like_match_smart_ppo")
    parser.add_argument("--num-players", type=int, choices=[2, 3, 4, 5], default=3)
    parser.add_argument(
        "--opponent",
        type=str,
        choices=["random", "heuristic", "strong_heuristic", "mixed_heuristic"],
        default="random",
        help=(
            "Opponent policy. mixed_heuristic keeps player 2 on the original "
            "heuristic and uses the stronger heuristic for player 3+."
        ),
    )
    parser.add_argument("--opponent-model-path", type=str, default=None,
                        help="Path to a previously trained model used as opponent (self-play). Falls back to random if missing.")
    parser.add_argument(
        "--opponent-model-seats",
        type=str,
        choices=["all", "last", "none"],
        default="all",
        help=(
            "Which opponent seats use --opponent-model-path. 'last' is cheaper "
            "and works well with mixed_heuristic: player 2 stays heuristic, "
            "player 3 uses the trained snapshot."
        ),
    )
    parser.add_argument("--discard-history-window", type=int, default=2)
    parser.add_argument(
        "--no-action-tactical-features",
        action="store_true",
        help=(
            "Disable the new AlphaLoba per-action tactical observation features. "
            "Use only for compatibility with old 799-dim models."
        ),
    )

    # Match-level reward params
    parser.add_argument("--target-points", type=int, default=100)
    parser.add_argument("--max-rounds-per-match", type=int, default=64)
    parser.add_argument("--match-win-bonus", type=float, default=150.0)
    parser.add_argument("--match-loss-penalty", type=float, default=150.0)
    parser.add_argument("--round-score-delta-coef", type=float, default=5.0)
    parser.add_argument("--round-win-terminal", type=float, default=10.0)
    parser.add_argument("--round-loss-terminal-coef", type=float, default=0.1)
    parser.add_argument("--max-reenganches", type=int, default=2,
                        help="Max reenganches per player per match (default 2). 0 disables reenganche.")

    # Round-level shaping (parent env)
    parser.add_argument("--reward-play-bonus", type=float, default=0.7)
    parser.add_argument("--reward-extend-bonus", type=float, default=0.5)
    parser.add_argument("--reward-cruzar-bonus", type=float, default=0.4)
    parser.add_argument("--reward-discard-with-play-penalty", type=float, default=1.1)
    parser.add_argument("--reward-discard-project-penalty", type=float, default=0.5)
    parser.add_argument("--reward-discard-low-before-high-penalty", type=float, default=0.2)
    parser.add_argument("--reward-discard-hot-card-penalty", type=float, default=0.25)
    parser.add_argument(
        "--reward-unsafe-meld-penalty",
        type=float,
        default=0.0,
        help=(
            "Penalty scale for laying a new meld while still holding enough hand "
            "points to bust if another player closes."
        ),
    )
    parser.add_argument(
        "--reward-discard-project-with-high-dead-penalty",
        type=float,
        default=0.0,
        help=(
            "Extra penalty when discarding a project card while a higher-value "
            "dead/non-project discard was available."
        ),
    )
    parser.add_argument(
        "--reward-discard-break-project-penalty",
        type=float,
        default=0.0,
        help="Extra penalty when the selected discard breaks the card's project status.",
    )
    parser.add_argument(
        "--reward-discard-high-dead-bonus",
        type=float,
        default=0.0,
        help=(
            "Small bonus for discarding the highest dead/non-project card while "
            "preserving some project in hand."
        ),
    )
    parser.add_argument("--no-round-shaping", action="store_true",
                        help="Zero out all 7 round-shaping coefficients (only match-level signal remains).")

    # Training infra
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--vec-env", type=str, choices=["dummy", "subproc"], default="dummy")
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--policy-net-arch", type=str, default="128,128",
                        help="Hidden layer widths for both policy and value MLPs (comma-separated). "
                             "Default 128,128 — larger than SB3 default (64,64) to handle the high "
                             "reward variance of match training.")
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default=None,
        help="Optional TensorBoard log directory, e.g. artifacts/tensorboard.",
    )

    # Callbacks
    parser.add_argument("--checkpoint-freq", type=int, default=100_000,
                        help="Total env-steps between checkpoint saves (split across n_envs).")
    parser.add_argument("--eval-freq", type=int, default=50_000,
                        help="Total env-steps between eval rollouts (split across n_envs).")
    parser.add_argument("--eval-episodes", type=int, default=10)

    # Resume / continuous training
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the latest checkpoint or best_model in the model-out directory.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Specific .zip path to resume from. Overrides --resume autodetection.")

    # Self-play snapshot pool
    parser.add_argument(
        "--self-play-snapshot-dir",
        type=str,
        default=None,
        help=(
            "If set, every --self-play-snapshot-freq env-steps the trainer saves the "
            "current model to this dir (FIFO-evicting beyond --self-play-pool-size). "
            "Each env reset() then samples a random snapshot from this dir as the "
            "opponent model — turning training into self-play against past selves."
        ),
    )
    parser.add_argument("--self-play-snapshot-freq", type=int, default=500_000,
                        help="Total env-steps between snapshot saves to the self-play pool.")
    parser.add_argument("--self-play-pool-size", type=int, default=10,
                        help="Max snapshots kept in the self-play pool (FIFO eviction).")
    parser.add_argument("--self-play-warmup-steps", type=int, default=0,
                        help="Skip self-play snapshot saves until this many timesteps have elapsed.")
    parser.add_argument("--self-play-sample-prob", type=float, default=1.0,
                        help="Per-match probability of sampling from the self-play pool (vs. using the base opponent). 1.0 = always self-play when pool is non-empty.")

    # Match JSON log
    parser.add_argument("--match-log-path", type=str, default=None,
                        help="If set, write the last N completed matches as JSON to this path "
                             "(rolling). Default: <model-out>_recent_matches.json")
    parser.add_argument("--match-log-keep", type=int, default=20,
                        help="How many of the most recent matches to keep in the JSON log. Default 20.")
    return parser.parse_args()


def _resolve_shaping(args: argparse.Namespace) -> dict[str, float]:
    if args.no_round_shaping:
        return {
            "reward_play_action_bonus": 0.0,
            "reward_extend_action_bonus": 0.0,
            "reward_cruzar_action_bonus": 0.0,
            "reward_discard_with_play_options_penalty": 0.0,
            "reward_discard_project_penalty": 0.0,
            "reward_discard_low_single_before_high_single_penalty": 0.0,
            "reward_discard_hot_card_penalty": 0.0,
            "reward_unsafe_meld_penalty": 0.0,
            "reward_discard_project_with_high_dead_penalty": 0.0,
            "reward_discard_break_project_penalty": 0.0,
            "reward_discard_high_dead_bonus": 0.0,
        }
    return {
        "reward_play_action_bonus": float(args.reward_play_bonus),
        "reward_extend_action_bonus": float(args.reward_extend_bonus),
        "reward_cruzar_action_bonus": float(args.reward_cruzar_bonus),
        "reward_discard_with_play_options_penalty": float(args.reward_discard_with_play_penalty),
        "reward_discard_project_penalty": float(args.reward_discard_project_penalty),
        "reward_discard_low_single_before_high_single_penalty": float(args.reward_discard_low_before_high_penalty),
        "reward_discard_hot_card_penalty": float(args.reward_discard_hot_card_penalty),
        "reward_unsafe_meld_penalty": float(args.reward_unsafe_meld_penalty),
        "reward_discard_project_with_high_dead_penalty": float(args.reward_discard_project_with_high_dead_penalty),
        "reward_discard_break_project_penalty": float(args.reward_discard_break_project_penalty),
        "reward_discard_high_dead_bonus": float(args.reward_discard_high_dead_bonus),
    }


def _make_env_fn(args: argparse.Namespace, rank: int):
    rules_kwargs = {"num_players": int(args.num_players)}
    shaping = _resolve_shaping(args)
    opp_path = args.opponent_model_path if args.opponent_model_seats != "none" else None

    def _init():
        rules = Rules(**rules_kwargs)
        # Each worker loads its own copy of the opponent model (subproc-safe).
        opp_model = None
        if opp_path:
            try:
                opp_model = load_model(opp_path)
            except Exception as exc:
                print(f"[worker {rank}] Failed to load opponent model {opp_path}: {exc}. Falling back to random.")
                opp_model = None
        env = RemoteLikeMatchSmartLobaEnv(
            rules=rules,
            seed=int(args.seed) + rank,
            opponent=args.opponent,
            discard_history_window=int(args.discard_history_window),
            target_points=int(args.target_points),
            max_rounds_per_match=int(args.max_rounds_per_match),
            round_score_delta_coef=float(args.round_score_delta_coef),
            match_win_bonus=float(args.match_win_bonus),
            match_loss_penalty=float(args.match_loss_penalty),
            round_win_terminal=float(args.round_win_terminal),
            round_loss_terminal_coef=float(args.round_loss_terminal_coef),
            max_reenganches=int(args.max_reenganches),
            reward_unsafe_meld_penalty=float(args.reward_unsafe_meld_penalty),
            reward_discard_project_with_high_dead_penalty=float(args.reward_discard_project_with_high_dead_penalty),
            reward_discard_high_dead_bonus=float(args.reward_discard_high_dead_bonus),
            reward_discard_break_project_penalty=float(args.reward_discard_break_project_penalty),
            enable_action_tactical_features=not bool(args.no_action_tactical_features),
            trained_opponent_model=opp_model,
            opponent_model_seats=args.opponent_model_seats,
            self_play_pool_dir=args.self_play_snapshot_dir,
            self_play_sample_prob=float(args.self_play_sample_prob),
        )
        for k, v in shaping.items():
            setattr(env, k, v)
        return env

    return _init


def _build_training_env(args: argparse.Namespace):
    n_envs = max(1, int(args.n_envs))
    fns = [_make_env_fn(args, i) for i in range(n_envs)]
    if n_envs == 1 or args.vec_env == "dummy":
        return DummyVecEnv(fns)
    return SubprocVecEnv(fns, start_method="spawn")


def _build_eval_env(args: argparse.Namespace):
    # Eval against random opponent (deterministic-ish baseline) regardless of training opponent.
    eval_args = argparse.Namespace(**{**vars(args), "opponent_model_path": None, "seed": int(args.seed) + 9999})
    return DummyVecEnv([_make_env_fn(eval_args, 0)])


def _force_close_vec_env(env, name: str = "env", soft_timeout: float = 3.0) -> None:
    """Close a (Subproc|Dummy)VecEnv reliably.

    SubprocVecEnv.close() calls remote.recv() to wait for each worker to finish, which
    deadlocks if a worker died mid-step from SIGINT. We try a clean close in a daemon
    thread; if it doesn't return within soft_timeout, we terminate/kill the workers.
    """
    import threading
    import time

    finished = threading.Event()

    def _closer() -> None:
        try:
            env.close()
        except Exception as exc:
            print(f"[shutdown] {name} close raised {type(exc).__name__}: {exc}")
        finally:
            finished.set()

    t = threading.Thread(target=_closer, daemon=True)
    t.start()
    finished.wait(timeout=soft_timeout)
    if finished.is_set():
        return

    print(f"[shutdown] {name} close() did not finish in {soft_timeout:.1f}s — forcing subprocess termination")
    procs = list(getattr(env, "processes", []) or [])
    for p in procs:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:
            pass
    time.sleep(0.3)
    for p in procs:
        try:
            if p.is_alive():
                p.kill()
            p.join(timeout=0.5)
        except Exception:
            pass


class MatchMetricsCallback(BaseCallback):
    """Logs match-level metrics aggregated over completed episodes."""

    def __init__(self, log_every: int = 5_000) -> None:
        super().__init__()
        self.log_every = int(log_every)
        self._wins: list[int] = []
        self._rounds: list[int] = []
        self._final_gap: list[float] = []
        self._eliminated_flags: list[int] = []
        self._elim_rounds: list[int] = []
        self._discard_project_with_high_dead: list[int] = []
        self._discard_breaks_project: list[int] = []
        self._discard_high_dead: list[int] = []
        self._discard_count = 0
        self._last_log_step = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if isinstance(info, dict):
                tactics = info.get("selected_action_tactics")
                if isinstance(tactics, dict) and bool(tactics.get("is_discard")):
                    self._discard_count += 1
                    self._discard_project_with_high_dead.append(
                        1 if bool(tactics.get("has_high_dead_alternative")) else 0
                    )
                    self._discard_breaks_project.append(
                        1 if bool(tactics.get("breaks_project")) else 0
                    )
                    card_points = int(tactics.get("card_points", 0) or 0)
                    best_dead = int(tactics.get("best_dead_card_points", 0) or 0)
                    is_project = bool(tactics.get("is_project_card"))
                    self._discard_high_dead.append(
                        1 if (not is_project and card_points >= best_dead and best_dead > 0) else 0
                    )
            if not done:
                continue
            if not isinstance(info, dict):
                continue
            scores = info.get("match_scores")
            rounds = info.get("rounds_played")
            winner = info.get("match_winner")
            if scores is None or rounds is None:
                continue
            scores_arr = np.asarray(scores, dtype=np.float32)
            eliminated = info.get("eliminated", [False] * scores_arr.size)
            if scores_arr.size == 0:
                continue
            if winner is not None:
                won = int(int(winner) == 0)
            else:
                # Fallback: compute winner ignoring eliminated players.
                active = np.array([not bool(e) for e in eliminated], dtype=bool)
                masked = np.where(active, scores_arr, np.inf)
                won = int(int(np.argmin(masked)) == 0)
            agent_score = float(scores_arr[0])
            best_other = float(np.min(scores_arr[1:])) if scores_arr.size > 1 else agent_score
            self._wins.append(won)
            self._rounds.append(int(rounds))
            self._final_gap.append(best_other - agent_score)
            agent_elim = bool(info.get("agent_eliminated", False))
            self._eliminated_flags.append(1 if agent_elim else 0)
            elim_round = info.get("agent_elimination_round")
            if agent_elim and isinstance(elim_round, int):
                self._elim_rounds.append(int(elim_round))

        if (self.num_timesteps - self._last_log_step) >= self.log_every and self._wins:
            self._last_log_step = self.num_timesteps
            n = len(self._wins)
            win_rate = float(np.mean(self._wins))
            avg_rounds = float(np.mean(self._rounds))
            avg_gap = float(np.mean(self._final_gap))
            elim_rate = float(np.mean(self._eliminated_flags)) if self._eliminated_flags else 0.0
            self.logger.record("match/win_rate", win_rate)
            self.logger.record("match/avg_rounds", avg_rounds)
            self.logger.record("match/avg_final_gap", avg_gap)
            self.logger.record("match/episodes_in_window", n)
            self.logger.record("match/agent_elimination_rate", elim_rate)
            if self._discard_count > 0:
                self.logger.record(
                    "tactics/discard_project_with_high_dead_rate",
                    float(np.mean(self._discard_project_with_high_dead)),
                )
                self.logger.record(
                    "tactics/discard_breaks_project_rate",
                    float(np.mean(self._discard_breaks_project)),
                )
                self.logger.record(
                    "tactics/discard_high_dead_rate",
                    float(np.mean(self._discard_high_dead)),
                )
                self.logger.record("tactics/discards_in_window", self._discard_count)
            if self._elim_rounds:
                self.logger.record(
                    "match/agent_avg_elimination_round", float(np.mean(self._elim_rounds))
                )
            self._wins.clear()
            self._rounds.clear()
            self._final_gap.clear()
            self._eliminated_flags.clear()
            self._elim_rounds.clear()
            self._discard_project_with_high_dead.clear()
            self._discard_breaks_project.clear()
            self._discard_high_dead.clear()
            self._discard_count = 0
        return True


class MatchWinRateEvalCallback(BaseCallback):
    """Eval callback that saves the best model based on WIN_RATE (not mean_reward).

    Better aligned with the deployment goal: a model that wins more matches against
    the heuristic is a better deployment candidate than one that maximizes shaping
    reward at the expense of actual wins.

    On each eval cycle:
      1. Run n_eval_episodes complete matches in eval_env (deterministic policy).
      2. Compute win_rate = fraction where match_winner == 0 (the agent).
      3. Save model when win_rate strictly improves; tie-break by mean_reward.
      4. Log eval/win_rate, eval/mean_reward, eval/avg_rounds, eval/n_episodes to TB.
    """

    def __init__(
        self,
        eval_env,
        best_path: str,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.best_path = Path(best_path)
        self.best_path.mkdir(parents=True, exist_ok=True)
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = max(1, int(n_eval_episodes))
        self.deterministic = bool(deterministic)
        self.best_win_rate: float = -1.0
        self.best_mean_reward_at_best: float = float("-inf")
        self._eval_history: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        wins, rewards, rounds, eliminated_flags, elim_rounds = self._run_eval()
        win_rate = float(np.mean(wins))
        mean_reward = float(np.mean(rewards))
        avg_rounds = float(np.mean(rounds)) if rounds else 0.0
        elim_rate = float(np.mean(eliminated_flags)) if eliminated_flags else 0.0
        avg_elim_round = float(np.mean(elim_rounds)) if elim_rounds else 0.0

        self.logger.record("eval/win_rate", win_rate)
        self.logger.record("eval/mean_reward", mean_reward)
        self.logger.record("eval/avg_rounds", avg_rounds)
        self.logger.record("eval/n_episodes", float(self.n_eval_episodes))
        self.logger.record("eval/agent_elimination_rate", elim_rate)
        if elim_rounds:
            self.logger.record("eval/agent_avg_elimination_round", avg_elim_round)

        improved = win_rate > self.best_win_rate or (
            win_rate == self.best_win_rate and mean_reward > self.best_mean_reward_at_best
        )
        if improved:
            self.best_win_rate = win_rate
            self.best_mean_reward_at_best = mean_reward
            self.model.save(str(self.best_path / "best_model"))
            if self.verbose > 0:
                print(
                    f"\n[eval] New best WIN_RATE: {win_rate:.3f}  "
                    f"(mean_reward={mean_reward:.1f}, avg_rounds={avg_rounds:.1f}, "
                    f"elim={elim_rate:.2f}, n={self.n_eval_episodes}, ts={self.num_timesteps})"
                )
        self._eval_history.append({
            "timestep": int(self.num_timesteps),
            "win_rate": win_rate,
            "mean_reward": mean_reward,
            "avg_rounds": avg_rounds,
            "agent_elimination_rate": elim_rate,
            "agent_avg_elimination_round": avg_elim_round if elim_rounds else None,
            "improved": bool(improved),
        })
        return True

    def _run_eval(
        self,
    ) -> tuple[list[int], list[float], list[int], list[int], list[int]]:
        from sb3_contrib.common.maskable.utils import get_action_masks
        wins: list[int] = []
        rewards: list[float] = []
        rounds: list[int] = []
        eliminated_flags: list[int] = []
        elim_rounds: list[int] = []
        obs = self.eval_env.reset()
        ep_reward = 0.0
        # Hard cap on actions to avoid an eval hang if a degenerate policy emerges.
        max_eval_steps = self.n_eval_episodes * 5000
        steps_taken = 0
        while len(wins) < self.n_eval_episodes and steps_taken < max_eval_steps:
            masks = get_action_masks(self.eval_env)
            action, _ = self.model.predict(
                obs, deterministic=self.deterministic, action_masks=masks
            )
            obs, reward, dones, infos = self.eval_env.step(action)
            ep_reward += float(reward[0])
            steps_taken += 1
            if bool(dones[0]):
                info = infos[0] if infos else {}
                ep_winner = info.get("match_winner")
                wins.append(1 if ep_winner == 0 else 0)
                rewards.append(ep_reward)
                rounds.append(int(info.get("rounds_played", 0)))
                agent_elim = bool(info.get("agent_eliminated", False))
                eliminated_flags.append(1 if agent_elim else 0)
                elim_round = info.get("agent_elimination_round")
                if agent_elim and isinstance(elim_round, int):
                    elim_rounds.append(int(elim_round))
                ep_reward = 0.0
                # VecEnv auto-resets, obs is already the next episode's first obs.
        return wins, rewards, rounds, eliminated_flags, elim_rounds


class MatchJsonLogger(BaseCallback):
    """Append the last N completed matches to a rolling JSON file on disk.

    Useful for offline inspection — feed the file path to the assistant when
    diagnosing what the model is doing wrong over recent episodes.
    """

    def __init__(self, log_path: str, max_entries: int = 20, hyperparams: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.log_path = Path(log_path)
        self.max_entries = max(1, int(max_entries))
        self.hyperparams = hyperparams or {}
        self._entries: list[dict[str, Any]] = self._load_existing()
        self._dirty = False
        self._last_flush_step = 0

    def _load_existing(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        try:
            text = self.log_path.read_text()
            data = json.loads(text)
            if isinstance(data, dict):
                entries = data.get("entries")
                if isinstance(entries, list):
                    return entries[-self.max_entries:]
            if isinstance(data, list):
                return data[-self.max_entries:]
        except Exception:
            pass
        return []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", []) or []
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done or not isinstance(info, dict):
                continue
            entry = {
                "ts": time.time(),
                "total_timesteps": int(self.num_timesteps),
                "match_winner": info.get("match_winner"),
                "match_scores": info.get("match_scores"),
                "rounds_played": info.get("rounds_played"),
                "eliminated": info.get("eliminated"),
                "agent_eliminated": info.get("agent_eliminated"),
                "agent_elimination_round": info.get("agent_elimination_round"),
                "reenganches_used": info.get("reenganches_used"),
                "n_active_players": info.get("n_active_players"),
                "round_records": info.get("round_records"),
                "selected_action_type": info.get("selected_action_type"),
                "phase_before_action": info.get("phase_before_action"),
                "had_play_options_before_action": info.get("had_play_options_before_action"),
                "selected_action_tactics": info.get("selected_action_tactics"),
            }
            self._entries.append(entry)
            self._dirty = True
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]
        # Flush every ~1024 steps OR when entries grew significantly to avoid disk thrashing.
        if self._dirty and (self.num_timesteps - self._last_flush_step) >= 1024:
            self._flush()
        return True

    def _flush(self) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_at": time.time(),
                "total_timesteps": int(self.num_timesteps),
                "max_entries": self.max_entries,
                "hyperparams": self.hyperparams,
                "entries": self._entries,
            }
            tmp = self.log_path.with_suffix(self.log_path.suffix + ".tmp")
            # Compact (no whitespace) — this file is consumed by another agent later
            # and there's no need to pretty-print it.
            tmp.write_text(json.dumps(payload, separators=(",", ":"), default=str))
            os.replace(tmp, self.log_path)
            self._dirty = False
            self._last_flush_step = self.num_timesteps
        except Exception:
            # Logging shouldn't break training; swallow errors.
            self._dirty = False

    def _on_training_end(self) -> None:
        if self._dirty:
            self._flush()


class SelfPlaySnapshotCallback(BaseCallback):
    """Periodically saves the current model to a self-play pool directory.

    Each subprocess env reads from this dir at reset() and samples a snapshot as
    its opponent — so the agent ends up training against a stream of its past
    selves. We keep at most `pool_size` snapshots, FIFO-evicting the oldest.
    """

    def __init__(
        self,
        pool_dir: str,
        save_freq_total: int,
        pool_size: int,
        warmup_steps: int = 0,
        n_envs: int = 1,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.pool_dir = Path(pool_dir)
        # Convert total env-steps → per-env steps; SB3 increments num_timesteps by n_envs each _on_step.
        self.save_freq_per_env = max(1, int(save_freq_total) // max(1, int(n_envs)))
        self.pool_size = max(1, int(pool_size))
        self.warmup_steps = max(0, int(warmup_steps))
        self._last_save_step = 0

    def _on_training_start(self) -> None:
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        # Anchor on current num_timesteps so resumed training doesn't immediately
        # save (we want to wait for the configured cadence after resume).
        self._last_save_step = int(self.num_timesteps)

    def _on_step(self) -> bool:
        steps_since_save = int(self.num_timesteps) - self._last_save_step
        if steps_since_save < self.save_freq_per_env:
            return True
        if int(self.num_timesteps) < self.warmup_steps:
            self._last_save_step = int(self.num_timesteps)
            return True
        self._save_snapshot()
        self._last_save_step = int(self.num_timesteps)
        return True

    def _save_snapshot(self) -> None:
        try:
            self.pool_dir.mkdir(parents=True, exist_ok=True)
            snap_path = self.pool_dir / f"snapshot_{int(self.num_timesteps):010d}.zip"
            self.model.save(str(snap_path))
            existing = sorted(self.pool_dir.glob("snapshot_*.zip"), key=lambda p: p.stat().st_mtime)
            while len(existing) > self.pool_size:
                victim = existing.pop(0)
                try:
                    victim.unlink()
                except Exception:
                    pass
            if self.verbose:
                print(f"[self-play] saved {snap_path.name} (pool size {len(existing)}/{self.pool_size})")
        except Exception as exc:
            print(f"[self-play] snapshot save failed: {exc}")


def _parse_net_arch(raw: str) -> list[int]:
    """Parse '128,128' → [128, 128]. Returns sane default on bad input."""
    try:
        widths = [int(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError:
        widths = []
    if not widths:
        return [128, 128]
    return [max(8, w) for w in widths]


def _resolve_resume_path(args: argparse.Namespace) -> str | None:
    """Decide which .zip to resume from, if any.

    Priority:
      1. --resume-from <path>: explicit path.
      2. --resume: autodetect — newest of best_model.zip / latest checkpoint / final model.
    """
    if args.resume_from:
        path = Path(args.resume_from)
        if path.exists():
            return str(path)
        print(f"[resume] --resume-from path does not exist: {path}; starting fresh.")
        return None
    if not args.resume:
        return None
    out_path = Path(args.model_out)
    candidates: list[Path] = []
    final_zip = out_path.with_suffix(".zip")
    if final_zip.exists():
        candidates.append(final_zip)
    best_dir = out_path.parent / f"{out_path.name}_best"
    best_zip = best_dir / "best_model.zip"
    if best_zip.exists():
        candidates.append(best_zip)
    ckpt_dir = out_path.parent / f"{out_path.name}_checkpoints"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
        if ckpts:
            candidates.append(ckpts[-1])
    if not candidates:
        print(f"[resume] No existing model found in {out_path.parent}; starting fresh.")
        return None
    # Pick the most recently modified.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def main() -> None:
    args = parse_args()

    train_env = _build_training_env(args)
    eval_env = _build_eval_env(args)

    n_envs = max(1, int(args.n_envs))
    n_steps = max(64, int(args.n_steps))
    batch_size = max(64, int(args.batch_size))
    rollout = n_steps * n_envs
    if rollout % batch_size != 0:
        # PPO requires batch_size to divide n_envs * n_steps.
        batch_size = max(64, rollout // max(1, rollout // batch_size))

    resume_path = _resolve_resume_path(args)
    if resume_path is not None:
        print(f"[resume] Loading existing model from {resume_path}")
        model = MaskablePPO.load(
            resume_path,
            env=train_env,
            tensorboard_log=args.tensorboard_log,
            custom_objects={
                # Allow re-tuning these without retraining from scratch.
                "learning_rate": float(args.learning_rate),
                "ent_coef": float(args.ent_coef),
            },
        )
        # Reset the env to make sure the loaded model uses the new env state.
        model.set_env(train_env)
    else:
        net_arch = _parse_net_arch(args.policy_net_arch)
        model = MaskablePPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=float(args.learning_rate),
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=float(args.ent_coef),
            policy_kwargs={"net_arch": net_arch},
            verbose=1,
            seed=int(args.seed),
            tensorboard_log=args.tensorboard_log,
        )

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.model_out).parent / f"{Path(args.model_out).name}_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir = Path(args.model_out).parent / f"{Path(args.model_out).name}_best"
    best_dir.mkdir(parents=True, exist_ok=True)

    match_log_path = args.match_log_path or str(
        Path(args.model_out).parent / f"{Path(args.model_out).name}_recent_matches.json"
    )
    hyperparams_snapshot = {
        "model_out": args.model_out,
        "policy_net_arch": _parse_net_arch(args.policy_net_arch),
        "num_players": args.num_players,
        "opponent": args.opponent,
        "opponent_model_path": args.opponent_model_path,
        "opponent_model_seats": args.opponent_model_seats,
        "action_tactical_features": not bool(args.no_action_tactical_features),
        "max_reenganches": args.max_reenganches,
        "ent_coef": args.ent_coef,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "match_win_bonus": args.match_win_bonus,
        "match_loss_penalty": args.match_loss_penalty,
        "round_score_delta_coef": args.round_score_delta_coef,
        "reward_play_bonus": args.reward_play_bonus,
        "reward_extend_bonus": args.reward_extend_bonus,
        "reward_cruzar_bonus": args.reward_cruzar_bonus,
        "reward_discard_with_play_penalty": args.reward_discard_with_play_penalty,
        "reward_discard_project_penalty": args.reward_discard_project_penalty,
        "reward_discard_low_before_high_penalty": args.reward_discard_low_before_high_penalty,
        "reward_discard_hot_card_penalty": args.reward_discard_hot_card_penalty,
        "reward_unsafe_meld_penalty": args.reward_unsafe_meld_penalty,
        "reward_discard_project_with_high_dead_penalty": args.reward_discard_project_with_high_dead_penalty,
        "reward_discard_break_project_penalty": args.reward_discard_break_project_penalty,
        "reward_discard_high_dead_bonus": args.reward_discard_high_dead_bonus,
        "no_round_shaping": args.no_round_shaping,
        "self_play_snapshot_dir": args.self_play_snapshot_dir,
        "self_play_snapshot_freq": args.self_play_snapshot_freq,
        "self_play_pool_size": args.self_play_pool_size,
        "self_play_warmup_steps": args.self_play_warmup_steps,
        "self_play_sample_prob": args.self_play_sample_prob,
    }

    callbacks: list[BaseCallback] = [
        CheckpointCallback(
            save_freq=max(1, int(args.checkpoint_freq) // n_envs),
            save_path=str(ckpt_dir),
            name_prefix=Path(args.model_out).name,
        ),
        MatchMetricsCallback(log_every=max(1, n_steps * n_envs)),
        MatchJsonLogger(
            log_path=match_log_path,
            max_entries=int(args.match_log_keep),
            hyperparams=hyperparams_snapshot,
        ),
        MatchWinRateEvalCallback(
            eval_env=eval_env,
            best_path=str(best_dir),
            eval_freq=max(1, int(args.eval_freq) // n_envs),
            n_eval_episodes=int(args.eval_episodes),
            deterministic=True,
        ),
    ]
    if args.self_play_snapshot_dir:
        callbacks.append(
            SelfPlaySnapshotCallback(
                pool_dir=args.self_play_snapshot_dir,
                save_freq_total=int(args.self_play_snapshot_freq),
                pool_size=int(args.self_play_pool_size),
                warmup_steps=int(args.self_play_warmup_steps),
                n_envs=n_envs,
                verbose=1,
            )
        )

    # When resuming, don't reset the global timestep counter — keep accumulating.
    reset_num_timesteps = resume_path is None
    try:
        model.learn(
            total_timesteps=int(args.timesteps),
            progress_bar=True,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=Path(args.model_out).name,
        )
    except KeyboardInterrupt:
        print("\n[interrupt] Caught Ctrl+C — saving current model state before exit...")
    finally:
        # Order matters: save first (most important), then ignore further SIGINTs so a
        # second Ctrl+C doesn't poison the cleanup with a half-shut-down state, then
        # force-close vec envs (subproc workers can deadlock on remote.recv after a
        # SIGINT mid-step — _force_close_vec_env terminates them if needed).
        import signal as _signal
        try:
            model.save(args.model_out)
        except Exception as save_exc:
            print(f"[shutdown] model.save failed: {save_exc}")
        _prev_sigint = _signal.signal(_signal.SIGINT, _signal.SIG_IGN)
        try:
            _force_close_vec_env(train_env, name="train_env")
            _force_close_vec_env(eval_env, name="eval_env")
        finally:
            _signal.signal(_signal.SIGINT, _prev_sigint)
        print(f"Match-aware model saved to {args.model_out}.zip")
        print(f"Checkpoints: {ckpt_dir}")
        print(f"Best model: {best_dir}/best_model.zip")
        print(f"Recent matches log: {match_log_path}")


if __name__ == "__main__":
    main()

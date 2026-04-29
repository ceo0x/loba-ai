"""Behavior cloning pretraining for the MaskablePPO Loba policy.

Loads a dataset of (obs, masks, actions) collected from the heuristic agent
(see collect_heuristic_data.py) and trains the policy net via masked
cross-entropy / NLL so it imitates the heuristic.

The output is a standard SB3 .zip that the main trainer can load with
`--resume-from <bc_model>.zip`.

Usage:
    .venv/bin/python -m loba_ai.training.pretrain_bc \\
        --dataset artifacts/bc_dataset_heuristic_3p.npz \\
        --num-players 3 \\
        --policy-net-arch 128,128 \\
        --epochs 10 --batch-size 1024 \\
        --output artifacts/loba_bc_pretrained.zip \\
        --seed 1
"""

from __future__ import annotations

import argparse
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from loba_ai.remote_like_match_smart_env import RemoteLikeMatchSmartLobaEnv
from loba_ai.rules import Rules


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Behavior cloning pretrain for MaskablePPO Loba policy")
    p.add_argument("--dataset", type=str, required=True, help=".npz from collect_heuristic_data")
    p.add_argument("--output", type=str, required=True, help="Output .zip (SB3 model)")
    p.add_argument("--num-players", type=int, choices=[2, 3, 4, 5], default=3)
    p.add_argument("--discard-history-window", type=int, default=2)
    p.add_argument("--target-points", type=int, default=100)
    p.add_argument("--max-reenganches", type=int, default=2)
    p.add_argument("--policy-net-arch", type=str, default="128,128")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--val-split", type=float, default=0.05, help="Fraction held out for validation.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device for BC training. Default auto picks cpu (matches MaskablePPO default).",
    )
    return p.parse_args()


def _parse_net_arch(raw: str) -> list[int]:
    return [int(x) for x in str(raw).split(",") if x.strip()]


def _make_env_for_shape(args: argparse.Namespace):
    """Returns a DummyVecEnv with one match env so MaskablePPO can build the policy
    with the right obs/action space. The env is used only for shape inference."""
    def _init():
        return RemoteLikeMatchSmartLobaEnv(
            rules=Rules(num_players=int(args.num_players)),
            seed=int(args.seed),
            opponent="random",
            discard_history_window=int(args.discard_history_window),
            target_points=int(args.target_points),
            max_reenganches=int(args.max_reenganches),
        )
    return DummyVecEnv([_init])


def main() -> None:
    args = parse_args()

    # ---- Load dataset ----
    ds_path = Path(args.dataset)
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset not found: {ds_path}")
    print(f"[bc] loading {ds_path} ({ds_path.stat().st_size/1e6:.1f} MB)")
    try:
        data = np.load(ds_path, allow_pickle=True)
        obs_all = data["obs"].astype(np.float32)
        masks_all = data["masks"].astype(np.int8)
        actions_all = data["actions"].astype(np.int64)
    except (OSError, EOFError, zipfile.BadZipFile, zlib.error, ValueError) as exc:
        raise RuntimeError(
            f"Could not read dataset {ds_path}. It is likely incomplete/corrupt "
            "because collection was interrupted while writing the .npz. "
            "Regenerate it with collect_heuristic_data."
        ) from exc
    n_total = obs_all.shape[0]
    obs_dim = obs_all.shape[1]
    action_space_n = masks_all.shape[1]
    print(f"[bc] dataset n={n_total}  obs_dim={obs_dim}  action_space={action_space_n}")
    if "meta" in data.files:
        try:
            meta = data["meta"].item()
            print(f"[bc] meta: {meta}")
        except Exception:
            pass

    # ---- Build a fresh MaskablePPO so the policy has matching obs/action space ----
    env = _make_env_for_shape(args)
    expected_obs_dim = int(env.observation_space.shape[0])
    expected_action_n = int(env.action_space.n)
    if expected_obs_dim != obs_dim or expected_action_n != action_space_n:
        raise ValueError(
            f"Dataset/env shape mismatch. Dataset obs_dim={obs_dim} action_space={action_space_n} "
            f"vs env obs_dim={expected_obs_dim} action_space={expected_action_n}. "
            f"Ensure --num-players, --discard-history-window match the collector args."
        )

    net_arch = _parse_net_arch(args.policy_net_arch)
    device = "cpu" if args.device == "auto" else args.device
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": dict(pi=net_arch, vf=net_arch)},
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        device=device,
        verbose=0,
    )

    policy = model.policy
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(args.learning_rate))

    # ---- Train/val split ----
    rng = np.random.default_rng(int(args.seed))
    perm = rng.permutation(n_total)
    n_val = int(max(1, args.val_split * n_total))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    print(f"[bc] split: train={len(train_idx)}  val={len(val_idx)}")

    obs_train = torch.from_numpy(obs_all[train_idx]).to(device)
    mask_train = torch.from_numpy(masks_all[train_idx]).to(device)
    act_train = torch.from_numpy(actions_all[train_idx]).to(device)
    obs_val = torch.from_numpy(obs_all[val_idx]).to(device)
    mask_val = torch.from_numpy(masks_all[val_idx]).to(device)
    act_val = torch.from_numpy(actions_all[val_idx]).to(device)

    n_train = obs_train.shape[0]
    batch_size = int(args.batch_size)

    # ---- Training loop ----
    print(f"[bc] starting training: epochs={args.epochs} batch_size={batch_size} lr={args.learning_rate}")
    t_start = time.time()
    for epoch in range(int(args.epochs)):
        epoch_start = time.time()
        policy.train()
        order = torch.randperm(n_train, device=device)
        total_loss = 0.0
        total_correct = 0
        total_count = 0
        for b_start in range(0, n_train, batch_size):
            b_end = min(b_start + batch_size, n_train)
            batch_ix = order[b_start:b_end]
            obs_b = obs_train[batch_ix]
            mask_b = mask_train[batch_ix].bool()
            act_b = act_train[batch_ix]

            distribution = policy.get_distribution(obs_b)
            distribution.apply_masking(mask_b)
            # log_prob of the heuristic's action under the (masked) distribution
            log_probs = distribution.log_prob(act_b)
            loss = -log_probs.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()

            total_loss += float(loss.detach()) * (b_end - b_start)
            # Argmax accuracy (under mask)
            with torch.no_grad():
                logits = distribution.distribution.logits  # (batch, action_n)
                pred = logits.argmax(dim=-1)
                total_correct += int((pred == act_b).sum())
                total_count += int(act_b.numel())

        train_loss = total_loss / max(1, n_train)
        train_acc = total_correct / max(1, total_count)

        # Validation
        policy.eval()
        with torch.no_grad():
            distribution = policy.get_distribution(obs_val)
            distribution.apply_masking(mask_val.bool())
            val_log_probs = distribution.log_prob(act_val)
            val_loss = float(-val_log_probs.mean())
            val_logits = distribution.distribution.logits
            val_pred = val_logits.argmax(dim=-1)
            val_acc = float((val_pred == act_val).float().mean())

        elapsed = time.time() - epoch_start
        total_elapsed = time.time() - t_start
        print(
            f"[bc] epoch {epoch+1:>2d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"epoch_time={elapsed:.1f}s  total={total_elapsed:.1f}s"
        )

    # ---- Save ----
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"[bc] saved {out_path}  size={out_path.stat().st_size/1e6:.1f} MB")
    print(f"[bc] usage: pass --resume-from {out_path} to train_remote_like_match_smart_rl")


if __name__ == "__main__":
    main()

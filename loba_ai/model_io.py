from __future__ import annotations

import numpy as np
from sb3_contrib import MaskablePPO


def save_model(model: MaskablePPO, path: str) -> None:
    model.save(path)


def load_model(path: str) -> MaskablePPO:
    return MaskablePPO.load(path)


def _is_no_match_obs_dim(dim: int) -> bool:
    return dim >= 115 and (dim - 111) % 2 == 0


def _is_match_obs_dim(dim: int) -> bool:
    return dim >= 118 and (dim - 114) % 2 == 0


def _no_match_players_from_dim(dim: int) -> int:
    return (dim - 111) // 2


def _match_players_from_dim(dim: int) -> int:
    return (dim - 114) // 2


def _parse_no_match_obs(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dim = int(obs.shape[0])
    players = _no_match_players_from_dim(dim)
    hand = obs[0:53]
    top_discard = obs[53:106]
    counts = obs[106 : 106 + players]
    opened = obs[106 + players : 106 + (2 * players)]
    stock_size = obs[106 + (2 * players) : 107 + (2 * players)]
    phase = obs[107 + (2 * players) : 110 + (2 * players)]
    legacy_pad = obs[110 + (2 * players) : 111 + (2 * players)]
    return hand, top_discard, counts, opened, stock_size, phase, legacy_pad


def _build_no_match_obs(
    hand: np.ndarray,
    top_discard: np.ndarray,
    counts: np.ndarray,
    opened: np.ndarray,
    stock_size: np.ndarray,
    phase: np.ndarray,
    legacy_pad: np.ndarray,
    target_players: int,
    dtype: np.dtype,
) -> np.ndarray:
    c = counts[:target_players]
    o = opened[:target_players]
    if c.shape[0] < target_players:
        c = np.concatenate([c, np.zeros(target_players - c.shape[0], dtype=dtype)])
    if o.shape[0] < target_players:
        o = np.concatenate([o, np.zeros(target_players - o.shape[0], dtype=dtype)])
    return np.concatenate([hand, top_discard, c, o, stock_size, phase, legacy_pad]).astype(dtype, copy=False)


def _adapt_observation_for_model(obs: np.ndarray, expected_obs_dim: int) -> np.ndarray:
    current_dim = int(obs.shape[0])
    if current_dim == expected_obs_dim:
        return obs

    if _is_no_match_obs_dim(current_dim) and _is_no_match_obs_dim(expected_obs_dim):
        hand, top_discard, counts, opened, stock_size, phase, legacy_pad = _parse_no_match_obs(obs)
        target_players = _no_match_players_from_dim(expected_obs_dim)
        return _build_no_match_obs(
            hand, top_discard, counts, opened, stock_size, phase, legacy_pad, target_players, obs.dtype
        )

    if _is_no_match_obs_dim(current_dim) and _is_match_obs_dim(expected_obs_dim):
        hand, top_discard, counts, opened, stock_size, phase, legacy_pad = _parse_no_match_obs(obs)
        target_players = _match_players_from_dim(expected_obs_dim)
        base = _build_no_match_obs(
            hand, top_discard, counts, opened, stock_size, phase, legacy_pad, target_players, obs.dtype
        )
        return np.concatenate([base, np.zeros(3, dtype=obs.dtype)]).astype(obs.dtype, copy=False)

    if _is_match_obs_dim(current_dim) and _is_no_match_obs_dim(expected_obs_dim):
        return _adapt_observation_for_model(obs[:-3], expected_obs_dim)

    if _is_match_obs_dim(current_dim) and _is_match_obs_dim(expected_obs_dim):
        adapted_no_match = _adapt_observation_for_model(obs[:-3], expected_obs_dim - 3)
        return np.concatenate([adapted_no_match, obs[-3:]]).astype(obs.dtype, copy=False)

    raise ValueError(
        f"Model observation mismatch: model expects {expected_obs_dim}, got {current_dim}. "
        "Use a model trained with a compatible environment/players configuration."
    )


def choose_action(model: MaskablePPO, obs: np.ndarray, action_mask: np.ndarray) -> int:
    expected_obs_shape = getattr(getattr(model, "observation_space", None), "shape", None)
    model_obs = obs
    if expected_obs_shape and len(expected_obs_shape) == 1:
        expected_obs_dim = int(expected_obs_shape[0])
        model_obs = _adapt_observation_for_model(obs, expected_obs_dim)

    action, _ = model.predict(model_obs, deterministic=True, action_masks=action_mask)
    return int(action)

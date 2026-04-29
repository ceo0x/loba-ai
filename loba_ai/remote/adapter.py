from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from loba_ai.model_io import choose_action
from loba_ai.remote_like_obs_builder import (
    SmartObsMemory,
    build_smart_match_obs,
    describe_action_tactics,
)

_SUIT_TO_OFFSET = {"C": 0, "D": 13, "H": 26, "S": 39}
_RANK_TO_VALUE = {
    "A": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
}
MAX_MELD_ACTIONS = 24
DISCARD_BASE_ACTION = 3 + MAX_MELD_ACTIONS
MAX_REMOTE_LIKE_ACTIONS = 256


def _card_token(card: dict[str, Any]) -> int:
    if card.get("joker"):
        return 52
    suit = card.get("suit")
    rank = card.get("rank")
    if suit not in _SUIT_TO_OFFSET or rank not in _RANK_TO_VALUE:
        return 52
    return _SUIT_TO_OFFSET[suit] + (_RANK_TO_VALUE[rank] - 1)


def _card_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("joker") == right.get("joker")
        and left.get("rank") == right.get("rank")
        and left.get("suit") == right.get("suit")
        and int(left.get("deck_id", -1)) == int(right.get("deck_id", -1))
    )


def _first_hand_index(hand: list[dict[str, Any]], card: dict[str, Any]) -> int | None:
    for idx, own in enumerate(hand):
        if _card_equal(own, card):
            return idx
    return None


def _phase_vector(phase: str) -> np.ndarray:
    vec = np.zeros(3, dtype=np.float32)
    mapping = {"draw": 0, "play_or_discard": 1, "cruzar_discard": 2}
    vec[mapping.get(phase, 0)] = 1.0
    return vec


def build_remote_obs_vector(observation: dict[str, Any]) -> np.ndarray:
    hand = np.zeros(53, dtype=np.float32)
    for card in observation.get("hand", []):
        hand[_card_token(card)] += 1.0
    hand /= 4.0

    top_discard = np.zeros(53, dtype=np.float32)
    discard = observation.get("discard_top")
    if isinstance(discard, dict):
        top_discard[_card_token(discard)] += 1.0

    my_seat = int(observation.get("seat", 0))
    num_players = int(observation.get("num_players", 2))
    num_players = max(2, num_players)
    other_sizes = observation.get("other_hand_sizes", [])
    laid = observation.get("has_laid_meld_this_round", [])
    seat_order = list(range(my_seat, num_players)) + list(range(0, my_seat))
    counts_values = []
    opened_values = []
    for seat in seat_order:
        size = float(other_sizes[seat]) if seat < len(other_sizes) else 0.0
        counts_values.append(size / 10.0)
        opened_values.append(1.0 if (seat < len(laid) and laid[seat]) else 0.0)
    counts = np.array(counts_values, dtype=np.float32)
    opened = np.array(opened_values, dtype=np.float32)

    stock = float(observation.get("stock_size", 0)) / 108.0
    phase = _phase_vector(str(observation.get("phase", "draw")))
    pad = np.zeros(1, dtype=np.float32)
    return np.concatenate([hand, top_discard, counts, opened, np.array([stock], dtype=np.float32), phase, pad]).astype(np.float32)


@dataclass(slots=True)
class CandidateAction:
    idx: int
    action: dict[str, Any]
    local_action_id: int | None
    score: float


def _map_remote_action_to_local_id(action: dict[str, Any], observation: dict[str, Any], meld_slot: int) -> int | None:
    action_type = action.get("type")
    if action_type == "DrawStock":
        return 0
    if action_type == "DrawDiscard":
        return 1
    if action_type in {"LayPierna", "LayEscalera", "ExtendMeld", "MoveJoker"}:
        if meld_slot >= MAX_MELD_ACTIONS:
            return None
        return 3 + meld_slot
    if action_type in {"Discard", "Cruzar"}:
        card = action.get("card")
        if not isinstance(card, dict):
            return None
        hand_index = _first_hand_index(observation.get("hand", []), card)
        if hand_index is None:
            return None
        return DISCARD_BASE_ACTION + hand_index
    return None


def _heuristic_score(action: dict[str, Any], observation: dict[str, Any]) -> float:
    action_type = action.get("type")
    if action_type == "DrawDiscard":
        return 6.0
    if action_type in {"LayEscalera", "LayPierna"}:
        cards = action.get("cards", [])
        return 5.0 + float(len(cards))
    if action_type in {"ExtendMeld", "MoveJoker"}:
        return 4.5
    if action_type == "Cruzar":
        return 4.0
    if action_type == "Discard":
        card = action.get("card", {})
        if card.get("joker"):
            # Keep joker when possible.
            return -4.0
        rank = _RANK_TO_VALUE.get(card.get("rank"), 0)
        return -float(rank)
    if action_type == "DrawStock":
        return 1.0
    return 0.0


def _consumed_cards_from_hand(action: dict[str, Any]) -> int:
    action_type = action.get("type")
    if action_type in {"LayPierna", "LayEscalera"}:
        return len(action.get("cards", []) or [])
    if action_type in {"ExtendMeld", "Discard", "Cruzar"}:
        return 1
    if action_type == "DrawDiscard":
        # DrawDiscard includes a mandatory play; it draws 1 and then consumes cards.
        # Net hand change in many cases is approximately consumed(play) - 1.
        nested = action.get("play")
        if isinstance(nested, dict):
            return max(0, _consumed_cards_from_hand(nested) - 1)
        return 0
    return 0


def _is_go_out_action(action: dict[str, Any], hand_size: int) -> bool:
    consumed = _consumed_cards_from_hand(action)
    return hand_size > 0 and consumed >= hand_size


def _tactics_for_meta(
    action: dict[str, Any],
    observation: dict[str, Any],
    obs_memory: SmartObsMemory | None,
) -> dict[str, Any]:
    seat = int(observation.get("seat", 0))
    current_score = 0.0
    target_points = 100
    if obs_memory is not None:
        try:
            current_score = float(obs_memory.match_scores[seat])
            target_points = int(obs_memory.target_points)
        except Exception:
            pass
    else:
        scores = observation.get("cumulative_scores") or []
        if seat < len(scores):
            try:
                current_score = float(scores[seat])
            except Exception:
                current_score = 0.0
    return describe_action_tactics(
        action,
        hand_payload=observation.get("hand", []) or [],
        current_score=current_score,
        target_points=target_points,
        table_melds=observation.get("melds_on_table") or [],
    )


def choose_remote_action(
    observation: dict[str, Any],
    model: object | None,
    remote_like_policy: bool = False,
    obs_memory: SmartObsMemory | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    legal_actions = observation.get("legal_actions", [])
    if not isinstance(legal_actions, list) or not legal_actions:
        raise ValueError("Observation has no legal_actions")

    # When the caller maintains session-level memory (matching what the training
    # env tracks), build the rich smart obs instead of the legacy 117-dim vector.
    # The smart obs vector matches the dimensionality of v15+ smart-match models
    # (≈543 dims for 3p) so model.predict gets the input it was trained on.
    if obs_memory is not None:
        smart_obs_vec = build_smart_match_obs(
            observation,
            obs_memory,
            melds=observation.get("melds_on_table"),
            legal_actions=observation.get("legal_actions"),
        )
        expected_shape = getattr(getattr(model, "observation_space", None), "shape", None)
        expected_dim = int(expected_shape[0]) if expected_shape else None
        if expected_dim is not None and expected_dim != int(smart_obs_vec.shape[0]):
            legacy_smart_obs_vec = build_smart_match_obs(
                observation,
                obs_memory,
                melds=observation.get("melds_on_table"),
                legal_actions=observation.get("legal_actions"),
                include_action_tactics=False,
            )
            if expected_dim == int(legacy_smart_obs_vec.shape[0]):
                smart_obs_vec = legacy_smart_obs_vec
    else:
        smart_obs_vec = None
    obs_vec = build_remote_obs_vector(observation)
    action_mask = np.zeros(2 + 1 + MAX_MELD_ACTIONS + 10, dtype=np.int8)
    hand_size = len(observation.get("hand", []))

    candidates: list[CandidateAction] = []
    meld_slot = 0
    for idx, action in enumerate(legal_actions):
        local_id = _map_remote_action_to_local_id(action, observation, meld_slot)
        if action.get("type") in {"LayPierna", "LayEscalera", "ExtendMeld", "MoveJoker"}:
            meld_slot += 1
        score = _heuristic_score(action, observation)
        candidates.append(CandidateAction(idx=idx, action=action, local_action_id=local_id, score=score))
        if local_id is not None and local_id < action_mask.shape[0]:
            action_mask[local_id] = 1

    go_out_candidates = [c for c in candidates if _is_go_out_action(c.action, hand_size)]
    go_out_available = len(go_out_candidates) > 0
    mapped_actions_total = int(np.sum(action_mask))
    action_type_counts: dict[str, int] = {}
    for c in candidates:
        action_type = str(c.action.get("type", "Unknown"))
        action_type_counts[action_type] = action_type_counts.get(action_type, 0) + 1

    used_model = False
    if remote_like_policy and model is not None:
        rl_mask = np.zeros(MAX_REMOTE_LIKE_ACTIONS, dtype=np.int8)
        for i in range(min(len(candidates), MAX_REMOTE_LIKE_ACTIONS)):
            rl_mask[i] = 1
        try:
            # Prefer the smart obs (rich features) when available, fall back to legacy.
            obs_for_model = smart_obs_vec if smart_obs_vec is not None else obs_vec
            predicted_idx = choose_action(model, obs_for_model, rl_mask)
            selected_idx = int(predicted_idx)
            if selected_idx < 0 or selected_idx >= len(candidates):
                selected_idx = 0
            best = candidates[selected_idx]
            used_model = True
            return best.action, {
                "selected_index": best.idx,
                "used_model": True,
                "remote_like_policy": True,
                "predicted_local_action_id": selected_idx,
                "fallback": False,
                "phase": observation.get("phase"),
                "legal_actions_total": len(candidates),
                "mapped_actions_total": mapped_actions_total,
                "go_out_actions_available": len(go_out_candidates),
                "go_out_available": go_out_available,
                "selected_go_out_action": _is_go_out_action(best.action, hand_size),
                "discard_selected": best.action.get("type") in {"Discard", "Cruzar"},
                "discard_while_go_out_available": go_out_available and best.action.get("type") in {"Discard", "Cruzar"},
                "action_type_selected": best.action.get("type"),
                "legal_action_type_counts": action_type_counts,
                "selected_action_tactics": _tactics_for_meta(best.action, observation, obs_memory),
            }
        except Exception as exc:
            best = max(candidates, key=lambda c: c.score)
            return best.action, {
                "selected_index": best.idx,
                "used_model": False,
                "remote_like_policy": True,
                "fallback": True,
                "fallback_reason": str(exc),
                "phase": observation.get("phase"),
                "legal_actions_total": len(candidates),
                "mapped_actions_total": mapped_actions_total,
                "go_out_actions_available": len(go_out_candidates),
                "go_out_available": go_out_available,
                "selected_go_out_action": _is_go_out_action(best.action, hand_size),
                "discard_selected": best.action.get("type") in {"Discard", "Cruzar"},
                "discard_while_go_out_available": go_out_available and best.action.get("type") in {"Discard", "Cruzar"},
                "action_type_selected": best.action.get("type"),
                "legal_action_type_counts": action_type_counts,
                "selected_action_tactics": _tactics_for_meta(best.action, observation, obs_memory),
            }

    if model is not None and int(action_mask.sum()) > 0:
        try:
            predicted_local_id = choose_action(model, obs_vec, action_mask)
            matching = [c for c in candidates if c.local_action_id == predicted_local_id]
            if matching:
                used_model = True
                best = max(matching, key=lambda c: c.score)
                return best.action, {
                    "selected_index": best.idx,
                    "used_model": True,
                    "predicted_local_action_id": predicted_local_id,
                    "fallback": False,
                    "phase": observation.get("phase"),
                    "legal_actions_total": len(candidates),
                    "mapped_actions_total": mapped_actions_total,
                    "go_out_actions_available": len(go_out_candidates),
                    "go_out_available": go_out_available,
                    "selected_go_out_action": _is_go_out_action(best.action, hand_size),
                    "discard_selected": best.action.get("type") in {"Discard", "Cruzar"},
                    "discard_while_go_out_available": go_out_available and best.action.get("type") in {"Discard", "Cruzar"},
                    "action_type_selected": best.action.get("type"),
                    "legal_action_type_counts": action_type_counts,
                    "selected_action_tactics": _tactics_for_meta(best.action, observation, obs_memory),
                }
        except Exception as exc:  # pragma: no cover - covered by fallback behavior assertions
            best = max(candidates, key=lambda c: c.score)
            return best.action, {
                "selected_index": best.idx,
                "used_model": False,
                "fallback": True,
                "fallback_reason": str(exc),
                "phase": observation.get("phase"),
                "legal_actions_total": len(candidates),
                "mapped_actions_total": mapped_actions_total,
                "go_out_actions_available": len(go_out_candidates),
                "go_out_available": go_out_available,
                "selected_go_out_action": _is_go_out_action(best.action, hand_size),
                "discard_selected": best.action.get("type") in {"Discard", "Cruzar"},
                "discard_while_go_out_available": go_out_available and best.action.get("type") in {"Discard", "Cruzar"},
                "action_type_selected": best.action.get("type"),
                "legal_action_type_counts": action_type_counts,
                "selected_action_tactics": _tactics_for_meta(best.action, observation, obs_memory),
            }

    best = max(candidates, key=lambda c: c.score)
    return best.action, {
        "selected_index": best.idx,
        "used_model": used_model,
        "fallback": True if model is not None else False,
        "phase": observation.get("phase"),
        "legal_actions_total": len(candidates),
        "mapped_actions_total": mapped_actions_total,
        "go_out_actions_available": len(go_out_candidates),
        "go_out_available": go_out_available,
        "selected_go_out_action": _is_go_out_action(best.action, hand_size),
        "discard_selected": best.action.get("type") in {"Discard", "Cruzar"},
        "discard_while_go_out_available": go_out_available and best.action.get("type") in {"Discard", "Cruzar"},
        "action_type_selected": best.action.get("type"),
        "legal_action_type_counts": action_type_counts,
        "selected_action_tactics": _tactics_for_meta(best.action, observation, obs_memory),
    }

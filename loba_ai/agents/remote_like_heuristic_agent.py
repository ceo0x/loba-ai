from __future__ import annotations

from typing import Any

import numpy as np

from loba_ai.remote_like_obs_builder import RUN_PROJECT_MAX_RANK_GAP


_RANK_VALUE = {"A": 11, "J": 10, "Q": 10, "K": 10}


def _rank_value(card_payload: dict) -> int:
    if bool(card_payload.get("joker")):
        return 15
    raw = str(card_payload.get("rank", "0"))
    if raw in _RANK_VALUE:
        return _RANK_VALUE[raw]
    try:
        return int(raw)
    except ValueError:
        return 0


def _rank_int(card_payload: dict) -> int:
    if bool(card_payload.get("joker")):
        return 99
    raw = str(card_payload.get("rank", "0"))
    rank_map = {"A": 1, "J": 11, "Q": 12, "K": 13}
    if raw in rank_map:
        return rank_map[raw]
    try:
        return int(raw)
    except ValueError:
        return 0


def _suit(card_payload: dict) -> str:
    return str(card_payload.get("suit", ""))


def _payload_key(card_payload: dict) -> str:
    if bool(card_payload.get("joker")):
        return f"J|{card_payload.get('deck_id')}"
    return f"{card_payload.get('rank')}|{card_payload.get('suit')}|{card_payload.get('deck_id')}"


def _is_project_card(card_payload: dict, hand_payload: list[dict]) -> bool:
    if bool(card_payload.get("joker")):
        return True
    rv = _rank_int(card_payload)
    if rv <= 0:
        return False
    suit = _suit(card_payload)
    target_key = _payload_key(card_payload)
    for c in hand_payload:
        if _payload_key(c) == target_key:
            continue
        if bool(c.get("joker")):
            continue
        rv2 = _rank_int(c)
        if rv2 == rv and _suit(c) != suit:
            return True
        rank_gap = abs(rv2 - rv)
        if _suit(c) == suit and 1 <= rank_gap <= RUN_PROJECT_MAX_RANK_GAP:
            return True
    return False


class RemoteLikeHeuristicAgent:
    """Rule-based opponent for the remote-like protocol.

    Priorities (in play_or_discard phase):
        1. Cruzar (free hand-size reduction; chainable).
        2. LayPierna / LayEscalera (largest meld first).
        3. ExtendMeld.
        4. MoveJoker — skipped unless it's the only legal action (avoids loops).
        5. Discard — pick the highest-value non-project, non-joker card.

    In draw phase: prefer DrawDiscard (it always embeds a productive play); else DrawStock.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    def act(
        self,
        legal_actions: list[dict[str, Any]],
        hand_payload: list[dict[str, Any]],
        phase: str,
    ) -> int:
        if not legal_actions:
            return 0

        if phase == "draw":
            return self._pick_draw(legal_actions)

        if phase == "play_or_discard":
            return self._pick_play_or_discard(legal_actions, hand_payload)

        # Unknown phase — fallback: first action.
        return 0

    # ---- per-phase selectors ---- #

    def _pick_draw(self, legal_actions: list[dict[str, Any]]) -> int:
        # DrawDiscard is only emitted when it consumes the top discard into a meld → always good.
        for i, a in enumerate(legal_actions):
            if str(a.get("type")) == "DrawDiscard":
                return i
        for i, a in enumerate(legal_actions):
            if str(a.get("type")) == "DrawStock":
                return i
        return 0

    def _pick_play_or_discard(
        self,
        legal_actions: list[dict[str, Any]],
        hand_payload: list[dict[str, Any]],
    ) -> int:
        # 1. Cruzar always — closes the round.
        for i, a in enumerate(legal_actions):
            if str(a.get("type")) == "Cruzar":
                return i

        # 2. Lay melds: largest first.
        best_lay_idx = -1
        best_lay_size = -1
        for i, a in enumerate(legal_actions):
            t = str(a.get("type"))
            if t in {"LayPierna", "LayEscalera"}:
                size = len(a.get("cards", [])) if isinstance(a.get("cards"), list) else 0
                if size > best_lay_size:
                    best_lay_size = size
                    best_lay_idx = i
        if best_lay_idx >= 0:
            return best_lay_idx

        # 3. ExtendMeld — pick first (any extension is good).
        for i, a in enumerate(legal_actions):
            if str(a.get("type")) == "ExtendMeld":
                return i

        # 4. Discard preferred over MoveJoker (avoids loops).
        discard_idx = self._pick_discard(legal_actions, hand_payload)
        for i, a in enumerate(legal_actions):
            if str(a.get("type")) == "Discard":
                return discard_idx

        # 5. Only MoveJoker available (rare edge case): take first.
        for i, a in enumerate(legal_actions):
            if str(a.get("type")) == "MoveJoker":
                return i

        return 0

    def _pick_discard(
        self,
        legal_actions: list[dict[str, Any]],
        hand_payload: list[dict[str, Any]],
    ) -> int:
        # Among Discard actions, prefer non-joker non-project cards, highest value first.
        candidates: list[tuple[int, int, int]] = []  # (priority, value, index)
        for i, a in enumerate(legal_actions):
            if str(a.get("type")) != "Discard":
                continue
            card = a.get("card", {})
            if not isinstance(card, dict):
                continue
            is_joker = bool(card.get("joker"))
            is_project = _is_project_card(card, hand_payload)
            value = _rank_value(card)
            # Lower priority = preferred. Joker is worst (3), project second (2), safe (1).
            if is_joker:
                priority = 3
            elif is_project:
                priority = 2
            else:
                priority = 1
            # Break ties by value: prefer HIGHER value (drop heavy cards first when safe).
            candidates.append((priority, -value, i))

        if candidates:
            candidates.sort()  # lowest priority, then most negative -value (i.e. highest value)
            return candidates[0][2]

        # No Discard in legal_actions (shouldn't happen in this phase, but defensive).
        return 0


class StrongRemoteLikeHeuristicAgent(RemoteLikeHeuristicAgent):
    """Stronger remote-like heuristic used as a tougher scripted opponent.

    Unlike ``RemoteLikeHeuristicAgent``, this selector scores every legal action.
    The main strategic difference is that it does not blindly cruzar first: it
    prefers plays that remove more cards/points, immediately go out, or preserve
    useful projects.
    """

    _PLAY_TYPES = {"LayPierna", "LayEscalera", "ExtendMeld", "MoveJoker", "Cruzar"}

    def _pick_draw(self, legal_actions: list[dict[str, Any]]) -> int:
        best_idx = 0
        best_score = -1_000_000.0
        for i, action in enumerate(legal_actions):
            action_type = str(action.get("type"))
            if action_type == "DrawStock":
                score = 0.0
            elif action_type == "DrawDiscard":
                nested = action.get("play", {})
                score = 8.0
                if isinstance(nested, dict):
                    score += self._score_play(nested, hand_payload=[], phase="draw")
            else:
                score = -100.0
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _pick_play_or_discard(
        self,
        legal_actions: list[dict[str, Any]],
        hand_payload: list[dict[str, Any]],
    ) -> int:
        best_idx = 0
        best_score = -1_000_000.0
        for i, action in enumerate(legal_actions):
            score = self._score_action(action, hand_payload, phase="play_or_discard")
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _score_action(
        self,
        action: dict[str, Any],
        hand_payload: list[dict[str, Any]],
        phase: str,
    ) -> float:
        action_type = str(action.get("type"))
        if action_type == "Discard":
            return self._score_discard(action, hand_payload)
        if action_type == "DrawStock":
            return 0.0
        if action_type == "DrawDiscard":
            nested = action.get("play", {})
            if not isinstance(nested, dict):
                return -100.0
            draw_score = 8.0 + self._score_play(nested, hand_payload, phase="draw")
            if self._draw_discard_goes_out(nested, hand_payload):
                draw_score += 10_000.0
            return draw_score
        return self._score_play(action, hand_payload, phase=phase)

    def _score_play(
        self,
        action: dict[str, Any],
        hand_payload: list[dict[str, Any]],
        phase: str,
    ) -> float:
        action_type = str(action.get("type"))
        if action_type not in self._PLAY_TYPES:
            return -100.0

        cards = self._action_cards(action)
        own_cards = self._matched_hand_cards(cards, hand_payload)
        cards_removed = len(own_cards) if hand_payload else len(cards)
        points_removed = sum(_rank_value(c) for c in own_cards) if own_cards else sum(_rank_value(c) for c in cards)
        score = 20.0 * cards_removed + 1.5 * points_removed

        if action_type in {"LayPierna", "LayEscalera"}:
            score += 18.0
            score += max(0, len(cards) - 3) * 8.0
        elif action_type == "ExtendMeld":
            score += 12.0
        elif action_type == "MoveJoker":
            score += 6.0
            score -= 10.0
        elif action_type == "Cruzar":
            # Good, but it does not end the turn and still requires a final discard.
            score += 4.0
            score -= 8.0

        if any(bool(c.get("joker")) for c in cards):
            score -= 12.0
        if phase == "play_or_discard" and action_type != "Cruzar" and self._play_goes_out(action, hand_payload):
            score += 10_000.0
        if hand_payload:
            remaining = max(0, len(hand_payload) - cards_removed)
            if remaining == 1 and action_type != "Cruzar":
                score += 35.0
            elif remaining == 2:
                score += 12.0
        return float(score)

    def _score_discard(self, action: dict[str, Any], hand_payload: list[dict[str, Any]]) -> float:
        card = action.get("card", {})
        if not isinstance(card, dict):
            return -100.0
        is_joker = bool(card.get("joker"))
        is_project = _is_project_card(card, hand_payload)
        value = _rank_value(card)
        score = 4.0 + (1.4 * value)
        if is_joker:
            score -= 70.0
        if is_project:
            score -= 32.0
        if self._is_low_single_before_high_single(card, hand_payload):
            score -= 12.0
        return float(score)

    def _action_cards(self, action: dict[str, Any]) -> list[dict[str, Any]]:
        action_type = str(action.get("type"))
        if action_type in {"LayPierna", "LayEscalera"}:
            cards = action.get("cards", [])
            return [c for c in cards if isinstance(c, dict)] if isinstance(cards, list) else []
        if action_type in {"ExtendMeld", "Cruzar", "Discard"}:
            card = action.get("card", {})
            return [card] if isinstance(card, dict) else []
        if action_type == "MoveJoker":
            card = action.get("replacement", {})
            return [card] if isinstance(card, dict) else []
        if action_type == "DrawDiscard":
            nested = action.get("play", {})
            return self._action_cards(nested) if isinstance(nested, dict) else []
        return []

    def _matched_hand_cards(
        self,
        action_cards: list[dict[str, Any]],
        hand_payload: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        remaining: dict[str, list[dict[str, Any]]] = {}
        for card in hand_payload:
            remaining.setdefault(_payload_key(card), []).append(card)

        matched: list[dict[str, Any]] = []
        for card in action_cards:
            bucket = remaining.get(_payload_key(card))
            if bucket:
                matched.append(bucket.pop())
        return matched

    def _play_goes_out(self, action: dict[str, Any], hand_payload: list[dict[str, Any]]) -> bool:
        if not hand_payload:
            return False
        if str(action.get("type")) == "Cruzar":
            return False
        own_cards = self._matched_hand_cards(self._action_cards(action), hand_payload)
        return len(own_cards) >= len(hand_payload)

    def _draw_discard_goes_out(
        self,
        nested_action: dict[str, Any],
        hand_payload: list[dict[str, Any]],
    ) -> bool:
        nested_type = str(nested_action.get("type"))
        if nested_type == "Cruzar":
            return False
        cards = self._action_cards(nested_action)
        own_cards = self._matched_hand_cards(cards, hand_payload)
        # DrawDiscard first adds the discard top, so a meld can consume all current
        # hand cards plus that newly drawn card.
        return len(own_cards) >= len(hand_payload) and len(cards) >= len(hand_payload) + 1

    def _is_low_single_before_high_single(
        self,
        card_payload: dict[str, Any],
        hand_payload: list[dict[str, Any]],
    ) -> bool:
        if _is_project_card(card_payload, hand_payload):
            return False
        selected_value = _rank_value(card_payload)
        singletons = [
            _rank_value(c)
            for c in hand_payload
            if not bool(c.get("joker")) and not _is_project_card(c, hand_payload)
        ]
        return bool(singletons) and selected_value < max(singletons)

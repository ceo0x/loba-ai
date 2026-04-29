from __future__ import annotations

from typing import Any

from loba_ai.cards import Card
from loba_ai.melds import find_all_melds, is_valid_run
from loba_ai.rules import Rules
from loba_ai.state import GameState

_ACTION_TYPE_PRIORITY = {
    "DrawStock": 0,
    "DrawDiscard": 1,
    "LayPierna": 2,
    "LayEscalera": 3,
    "ExtendMeld": 4,
    "MoveJoker": 5,
    "Cruzar": 6,
    "Discard": 7,
}


def _card_sort_key(card_payload: dict[str, Any]) -> tuple:
    if bool(card_payload.get("joker")):
        return (1, 99, "J", int(card_payload.get("deck_id", -1)))
    rank_raw = str(card_payload.get("rank", "0"))
    rank_map = {"A": 1, "J": 11, "Q": 12, "K": 13}
    try:
        rank_num = int(rank_raw)
    except ValueError:
        rank_num = rank_map.get(rank_raw, 0)
    return (0, rank_num, str(card_payload.get("suit", "")), int(card_payload.get("deck_id", -1)))


def _action_sort_key(action: dict[str, Any]) -> tuple:
    action_type = str(action.get("type", "Unknown"))
    priority = _ACTION_TYPE_PRIORITY.get(action_type, 99)
    if action_type in {"LayPierna", "LayEscalera"}:
        cards = action.get("cards", [])
        if isinstance(cards, list):
            key_cards = tuple(sorted((_card_sort_key(c) for c in cards if isinstance(c, dict))))
        else:
            key_cards = ()
        return (priority, key_cards)
    if action_type in {"Discard", "Cruzar"}:
        card = action.get("card")
        if isinstance(card, dict):
            return (priority, _card_sort_key(card))
        return (priority, ())
    if action_type in {"ExtendMeld", "MoveJoker"}:
        meld_id = int(action.get("meld_id", -1))
        card = action.get("card") or action.get("replacement")
        card_key = _card_sort_key(card) if isinstance(card, dict) else ()
        return (priority, meld_id, card_key)
    if action_type == "DrawDiscard":
        nested = action.get("play")
        nested_key = _action_sort_key(nested) if isinstance(nested, dict) else ()
        return (priority, nested_key)
    return (priority,)


def canonicalize_remote_like_legal_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(actions, key=_action_sort_key)


def _card_ref(card: Card, hand_index: int | None = None) -> dict[str, Any]:
    if card.is_joker:
        out: dict[str, Any] = {"joker": True, "deck_id": card.deck_id}
    else:
        rank_map = {1: "A", 11: "J", 12: "Q", 13: "K"}
        suit_map = {"spades": "S", "hearts": "H", "clubs": "C", "diamonds": "D"}
        out = {"rank": rank_map.get(card.rank, str(card.rank)), "suit": suit_map.get(card.suit or "clubs", "C"), "deck_id": card.deck_id}
    if hand_index is not None:
        out["hand_index"] = hand_index
    return out


def _find_hand_index(hand: list[Card], target: Card) -> int | None:
    for i, c in enumerate(hand):
        if c is target:
            return i
    return None


def _find_top_discard_consuming_plays(
    hand: list[Card],
    top_discard: Card,
    rules: Rules,
    max_meld_actions: int,
    table_melds: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Plays that consume ``top_discard`` immediately (required by DrawDiscard).

    Includes:
      - LayPierna / LayEscalera: forming a new meld with the picked card.
      - ExtendMeld: appending the picked card to an existing meld.
      - MoveJoker: using the picked card to replace an end-joker on a run.
      - Cruzar: matching the picked card against an existing pierna (rank+suit).
    """
    out: list[dict[str, Any]] = []
    simulated = hand + [top_discard]
    for meld in find_all_melds(simulated, rules, max_results=max_meld_actions):
        if top_discard not in meld.cards:
            continue
        kind = "LayPierna" if meld.kind == "group" else "LayEscalera"
        cards_payload = [_card_ref(c) for c in meld.cards]
        out.append({"type": kind, "cards": cards_payload})

    if not table_melds or top_discard.is_joker:
        # Joker-as-discard cannot extend/cruzar (only used inside a fresh meld).
        # If a joker IS at the top, the LayEscalera path above already covers it.
        return out

    # ExtendMeld with the picked card.
    for meld in table_melds:
        meld_cards = meld.get("cards", [])
        kind = meld.get("kind")
        if kind == "escalera":
            trial = list(meld_cards) + [top_discard]
            if is_valid_run(trial, rules):
                out.append(
                    {
                        "type": "ExtendMeld",
                        "meld_id": meld.get("meld_id"),
                        "card": _card_ref(top_discard),
                    }
                )
        elif kind == "pierna":
            naturals = [c for c in meld_cards if not c.is_joker]
            if not naturals:
                continue
            rank = naturals[0].rank
            suits_used = {c.suit for c in naturals}
            if top_discard.rank == rank and top_discard.suit in suits_used:
                out.append(
                    {
                        "type": "ExtendMeld",
                        "meld_id": meld.get("meld_id"),
                        "card": _card_ref(top_discard),
                    }
                )

    # MoveJoker using the picked card as the replacement.
    for meld in table_melds:
        if meld.get("kind") != "escalera":
            continue
        cards = meld.get("cards", [])
        if len(cards) < 2:
            continue
        meld_id = meld.get("meld_id")
        meld_suit = next((c.suit for c in cards if not c.is_joker), None)
        if meld_suit is None or meld_suit != top_discard.suit:
            continue
        first_natural = next((c for c in cards if not c.is_joker), None)
        last_natural = next((c for c in reversed(cards) if not c.is_joker), None)
        if first_natural is None or last_natural is None:
            continue
        if cards[0].is_joker:
            joker_rank = first_natural.rank - 1
            new_joker_rank = last_natural.rank + 1
            if (
                top_discard.rank == joker_rank
                and 1 <= joker_rank <= 13
                and 1 <= new_joker_rank <= 13
            ):
                out.append(
                    {
                        "type": "MoveJoker",
                        "meld_id": meld_id,
                        "replacement": _card_ref(top_discard),
                    }
                )
        if cards[-1].is_joker:
            joker_rank = last_natural.rank + 1
            new_joker_rank = first_natural.rank - 1
            if (
                top_discard.rank == joker_rank
                and 1 <= joker_rank <= 13
                and 1 <= new_joker_rank <= 13
            ):
                out.append(
                    {
                        "type": "MoveJoker",
                        "meld_id": meld_id,
                        "replacement": _card_ref(top_discard),
                    }
                )

    # Cruzar with the picked card. Hand must have at least 1 more card to satisfy
    # the post-cruzar discard requirement (the picked card itself goes to discard
    # via cruzar; we still need another card for the mandatory final discard).
    if len(hand) >= 1:
        pierna_keys: set[tuple[int, str | None]] = set()
        for meld in table_melds:
            if meld.get("kind") != "pierna":
                continue
            for c in meld.get("cards", []):
                if c.is_joker:
                    continue
                pierna_keys.add((c.rank, c.suit))
        if (top_discard.rank, top_discard.suit) in pierna_keys:
            out.append({"type": "Cruzar", "card": _card_ref(top_discard)})

    return out


def _extend_candidates_for_runs(hand: list[Card], table_melds: list[dict[str, Any]], rules: Rules) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for meld in table_melds:
        meld_cards = meld.get("cards", [])
        kind = meld.get("kind")
        if kind == "escalera":
            for idx, card in enumerate(hand):
                trial = list(meld_cards) + [card]
                if is_valid_run(trial, rules):
                    out.append({"type": "ExtendMeld", "meld_id": meld.get("meld_id"), "card": _card_ref(card, hand_index=idx)})
        elif kind == "pierna":
            naturals = [c for c in meld_cards if not c.is_joker]
            if not naturals:
                continue
            rank = naturals[0].rank
            suits_used = {c.suit for c in naturals}
            for idx, card in enumerate(hand):
                if card.is_joker:
                    continue
                if card.rank != rank:
                    continue
                # Brother sample indicates extending pierna with same-rank card and a suit
                # already present (typically 2nd deck copy).
                if card.suit in suits_used:
                    out.append({"type": "ExtendMeld", "meld_id": meld.get("meld_id"), "card": _card_ref(card, hand_index=idx)})
    return out


def _move_joker_candidates(hand: list[Card], table_melds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strict MoveJoker candidates per Loba rules:

    - Only on escaleras (runs).
    - Joker must be at left or right end (not the middle).
    - Replacement must be the EXACT rank the joker represents AND same suit as the meld.
    - Post-move, the joker is shifted to the opposite end; its new rank must be in [1, 13].
    """
    out: list[dict[str, Any]] = []
    for meld in table_melds:
        if meld.get("kind") != "escalera":
            continue
        cards: list[Card] = meld.get("cards", [])
        if len(cards) < 2:
            continue
        meld_id = meld.get("meld_id")
        meld_suit = next((c.suit for c in cards if not c.is_joker), None)
        if meld_suit is None:
            continue

        # LEFT joker: represents the rank just before the leftmost natural.
        if cards[0].is_joker:
            first_natural = next((c for c in cards if not c.is_joker), None)
            last_natural = next((c for c in reversed(cards) if not c.is_joker), None)
            if first_natural is not None and last_natural is not None:
                joker_rank = first_natural.rank - 1
                # After move, joker shifts to the right end → new joker rank = last_natural + 1.
                new_joker_rank = last_natural.rank + 1
                if 1 <= joker_rank <= 13 and 1 <= new_joker_rank <= 13:
                    for idx, card in enumerate(hand):
                        if card.is_joker:
                            continue
                        if card.suit == meld_suit and card.rank == joker_rank:
                            out.append(
                                {
                                    "type": "MoveJoker",
                                    "meld_id": meld_id,
                                    "replacement": _card_ref(card, hand_index=idx),
                                }
                            )

        # RIGHT joker: represents the rank just after the rightmost natural.
        # Skip if both ends are jokers and we already proposed the left-side candidate
        # for the same hand card (avoid duplicate proposals on rare both-end-joker melds).
        if cards[-1].is_joker:
            first_natural = next((c for c in cards if not c.is_joker), None)
            last_natural = next((c for c in reversed(cards) if not c.is_joker), None)
            if first_natural is not None and last_natural is not None:
                joker_rank = last_natural.rank + 1
                new_joker_rank = first_natural.rank - 1
                if 1 <= joker_rank <= 13 and 1 <= new_joker_rank <= 13:
                    for idx, card in enumerate(hand):
                        if card.is_joker:
                            continue
                        if card.suit == meld_suit and card.rank == joker_rank:
                            out.append(
                                {
                                    "type": "MoveJoker",
                                    "meld_id": meld_id,
                                    "replacement": _card_ref(card, hand_index=idx),
                                }
                            )
    return out


def _cruzar_candidates(hand: list[Card], table_melds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # Cruzar removes a card from hand without ending the turn; the player still
    # owes a final Discard. Require at least 2 cards so a final discard is feasible.
    if len(hand) < 2:
        return out
    pierna_cards: list[tuple[int, str | None]] = []
    for meld in table_melds:
        if meld.get("kind") != "pierna":
            continue
        for c in meld.get("cards", []):
            if c.is_joker:
                continue
            pierna_cards.append((c.rank, c.suit))
    if not pierna_cards:
        return out
    for idx, card in enumerate(hand):
        if card.is_joker:
            continue
        if (card.rank, card.suit) in pierna_cards:
            out.append({"type": "Cruzar", "card": _card_ref(card, hand_index=idx)})
    return out


def build_remote_like_legal_actions(
    state: GameState,
    rules: Rules,
    table_melds: list[dict[str, Any]],
    max_meld_actions: int = 24,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    player = state.players[state.current_player]
    hand = player.hand
    phase = state.phase

    if phase == "draw":
        actions.append({"type": "DrawStock"})
        if state.discard_pile:
            top_discard = state.discard_pile[-1]
            plays = _find_top_discard_consuming_plays(
                hand, top_discard, rules, max_meld_actions, table_melds=table_melds
            )
            for p in plays:
                actions.append({"type": "DrawDiscard", "play": p})
        return actions

    if phase == "play_or_discard":
        for meld in find_all_melds(hand, rules, max_results=max_meld_actions):
            kind = "LayPierna" if meld.kind == "group" else "LayEscalera"
            actions.append({"type": kind, "cards": [_card_ref(c) for c in meld.cards]})
        actions.extend(_extend_candidates_for_runs(hand, table_melds, rules))
        actions.extend(_move_joker_candidates(hand, table_melds))
        actions.extend(_cruzar_candidates(hand, table_melds))
        for i, c in enumerate(hand):
            if c.is_joker and len(hand) > 1:
                continue
            actions.append({"type": "Discard", "card": _card_ref(c, hand_index=i)})
        return actions

    return actions

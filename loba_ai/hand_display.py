from __future__ import annotations

from collections.abc import Iterable

from loba_ai.cards import Card
from loba_ai.melds import Meld, find_all_melds
from loba_ai.rules import Rules

_SUIT_ORDER = {"clubs": 0, "diamonds": 1, "hearts": 2, "spades": 3}
_RANK_LABEL = {1: "A", 11: "J", 12: "Q", 13: "K"}
_MELD_SCAN_LIMIT = 64


def card_label(card: Card) -> str:
    if card.is_joker:
        return "Joker"
    rank = _RANK_LABEL.get(card.rank, str(card.rank))
    suit_map = {"clubs": "C", "diamonds": "D", "hearts": "H", "spades": "S"}
    return f"{rank}{suit_map[card.suit]}"


def _fallback_card_sort_key(card: Card) -> tuple[int, int]:
    if card.is_joker:
        return (99, 99)
    return (_SUIT_ORDER.get(card.suit or "clubs", 9), card.rank)


def _meld_card_sort_key(card: Card) -> tuple[int, int, int]:
    if card.is_joker:
        return (1, 99, 99)
    return (0, _SUIT_ORDER.get(card.suit or "clubs", 9), card.rank)


def _map_meld_to_indices(hand: list[Card], meld_cards: Iterable[Card]) -> list[int] | None:
    indices: list[int] = []
    for card in meld_cards:
        hand_index = next((i for i, hand_card in enumerate(hand) if hand_card is card), None)
        if hand_index is None:
            return None
        indices.append(hand_index)
    return indices


def build_hand_display(hand: list[Card], rules: Rules) -> list[dict]:
    """Return display-first hand order without mutating game state.

    The result preserves original hand references through ``hand_index`` while grouping
    likely meld blocks first and then remaining cards by suit/rank.
    """
    candidates: list[tuple[int, list[int], Meld]] = []
    for meld in find_all_melds(hand, rules, max_results=_MELD_SCAN_LIMIT):
        mapped = _map_meld_to_indices(hand, meld.cards)
        if mapped is None:
            continue
        candidates.append((len(mapped), mapped, meld))

    candidates.sort(key=lambda x: x[0], reverse=True)

    used_indices: set[int] = set()
    ordered_indices: list[int] = []
    for _, meld_indices, meld in candidates:
        if any(i in used_indices for i in meld_indices):
            continue
        sorted_meld_indices = sorted(meld_indices, key=lambda i: _meld_card_sort_key(hand[i]))
        ordered_indices.extend(sorted_meld_indices)
        used_indices.update(sorted_meld_indices)

    remaining = [i for i in range(len(hand)) if i not in used_indices]
    remaining.sort(key=lambda i: _fallback_card_sort_key(hand[i]))
    ordered_indices.extend(remaining)

    return [{"label": card_label(hand[i]), "hand_index": i} for i in ordered_indices]

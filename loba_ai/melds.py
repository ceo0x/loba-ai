from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from loba_ai.cards import Card
from loba_ai.rules import Rules


@dataclass(slots=True)
class Meld:
    cards: tuple[Card, ...]
    kind: str  # "group" | "run"


def is_valid_group(cards: list[Card], rules: Rules) -> bool:
    if len(cards) < rules.min_group_size:
        return False
    jokers = [c for c in cards if c.is_joker]
    naturals = [c for c in cards if not c.is_joker]
    if not naturals:
        return False

    rank = naturals[0].rank
    if any(c.rank != rank for c in naturals):
        return False

    if not rules.allow_duplicate_suits_in_group:
        suits = [c.suit for c in naturals]
        if len(set(suits)) != len(suits):
            return False

    if (not rules.allow_jokers and jokers) or (jokers and not rules.allow_jokers_in_group):
        return False

    return True


def is_valid_run(cards: list[Card], rules: Rules) -> bool:
    if len(cards) < rules.min_run_size:
        return False

    jokers = [c for c in cards if c.is_joker]
    naturals = [c for c in cards if not c.is_joker]
    if not naturals:
        return False

    suit = naturals[0].suit
    if any(c.suit != suit for c in naturals):
        return False

    if (not rules.allow_jokers and jokers) or (jokers and not rules.allow_jokers_in_run):
        return False

    ranks = sorted(c.rank for c in naturals)
    if len(set(ranks)) != len(ranks):
        return False

    gaps = 0
    max_internal_gap = 0
    for i in range(1, len(ranks)):
        diff = ranks[i] - ranks[i - 1]
        if diff <= 0:
            return False
        gap_here = diff - 1
        gaps += gap_here
        if gap_here > max_internal_gap:
            max_internal_gap = gap_here

    if gaps > len(jokers):
        return False

    # Loba house rule: no more than 2 jokers in a row anywhere in the run.
    if max_internal_gap > 2:
        return False

    # Jokers not consumed by internal gaps must sit at the ends. Each end can
    # hold at most 2 jokers (same rule), bounded by 1..13 ranks.
    extreme_jokers = len(jokers) - gaps
    if extreme_jokers < 0:
        return False
    left_room = max(0, ranks[0] - 1)
    right_room = max(0, 13 - ranks[-1])
    for left_count in range(min(2, left_room) + 1):
        right_count = extreme_jokers - left_count
        if 0 <= right_count <= min(2, right_room):
            return True
    return False


def is_valid_meld(cards: list[Card], rules: Rules) -> bool:
    return is_valid_group(cards, rules) or is_valid_run(cards, rules)


def find_all_melds(hand: list[Card], rules: Rules, max_results: int = 24) -> list[Meld]:
    found: list[Meld] = []
    for size in range(rules.min_group_size, len(hand) + 1):
        for combo in combinations(hand, size):
            as_list = list(combo)
            if is_valid_group(as_list, rules):
                found.append(Meld(cards=tuple(combo), kind="group"))
            elif is_valid_run(as_list, rules):
                found.append(Meld(cards=tuple(combo), kind="run"))
            if len(found) >= max_results:
                return found
    return found


def discard_take_melds(hand: list[Card], top_discard: Card, rules: Rules, max_results: int = 24) -> list[Meld]:
    """Melds one could play immediately after picking ``top_discard`` (same object) into ``hand``.

    Each returned meld includes ``top_discard`` and is valid under ``rules``.
    """
    simulated = hand + [top_discard]
    all_m = find_all_melds(simulated, rules, max_results=max_results)
    return [m for m in all_m if top_discard in m.cards]


def can_open_with_contract(hand: list[Card], rules: Rules, round_index: int = 0) -> bool:
    contract = rules.contracts[min(round_index, len(rules.contracts) - 1)]
    melds = find_all_melds(hand, rules, max_results=50)
    groups = sum(1 for m in melds if m.kind == "group")
    runs = sum(1 for m in melds if m.kind == "run")
    return groups >= contract.groups and runs >= contract.runs

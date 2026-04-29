from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

SUITS = ("clubs", "diamonds", "hearts", "spades")
RANKS = tuple(range(1, 14))


@dataclass(frozen=True, slots=True)
class Card:
    rank: int
    suit: str | None
    deck_id: int
    is_joker: bool = False

    @property
    def token(self) -> int:
        if self.is_joker:
            return 52
        suit_ix = SUITS.index(self.suit or "clubs")
        return suit_ix * 13 + (self.rank - 1)

    @property
    def points(self) -> int:
        if self.is_joker:
            return 15
        if self.rank == 1:
            return 11
        if self.rank >= 11:
            return 10
        return self.rank


def deck(num_decks: int = 2, num_jokers: int = 4) -> list[Card]:
    cards: list[Card] = []
    for d in range(num_decks):
        for suit in SUITS:
            for rank in RANKS:
                cards.append(Card(rank=rank, suit=suit, deck_id=d, is_joker=False))
    for j in range(num_jokers):
        cards.append(Card(rank=0, suit=None, deck_id=j, is_joker=True))
    return cards


def hand_points(cards: Iterable[Card]) -> int:
    return sum(c.points for c in cards)

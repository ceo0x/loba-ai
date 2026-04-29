from __future__ import annotations

from dataclasses import dataclass, field

from loba_ai.cards import Card


@dataclass(slots=True)
class PlayerState:
    hand: list[Card] = field(default_factory=list)
    melds: list[tuple[Card, ...]] = field(default_factory=list)
    has_opened: bool = False
    score: int = 0


@dataclass(slots=True)
class GameState:
    players: list[PlayerState]
    current_player: int
    stock_pile: list[Card]
    discard_pile: list[Card]
    melds_on_table: list[tuple[Card, ...]]
    round_index: int
    turn_number: int
    finished: bool = False
    winner: int | None = None
    phase: str = "draw"  # draw -> meld -> discard

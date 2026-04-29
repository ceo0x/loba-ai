from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Contract:
    groups: int = 0
    runs: int = 0


@dataclass(slots=True)
class Rules:
    num_players: int = 2
    num_decks: int = 2
    num_jokers: int = 4
    cards_per_player: int = 9
    min_group_size: int = 3
    min_run_size: int = 3
    allow_duplicate_suits_in_group: bool = False
    allow_jokers: bool = True
    allow_jokers_in_group: bool = False
    allow_jokers_in_run: bool = True
    ace_low: bool = True
    ace_high: bool = False
    must_discard_to_go_out: bool = True
    # If True: may only take discard when a meld exists that includes the top discard card;
    # that meld is played immediately and the turn goes to the discard phase.
    must_meld_if_draw_discard: bool = True
    max_round_turns: int = 300
    contracts: list[Contract] = field(default_factory=lambda: [Contract(groups=1, runs=0)])

    @property
    def max_hand_size(self) -> int:
        return self.cards_per_player + 1

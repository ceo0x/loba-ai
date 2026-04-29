from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from loba_ai.cards import Card, deck, hand_points
from loba_ai.melds import discard_take_melds, find_all_melds
from loba_ai.rules import Rules
from loba_ai.state import GameState, PlayerState


@dataclass(slots=True)
class StepResult:
    state: GameState
    reward: float
    done: bool
    info: dict


class LobaGameEngine:
    def __init__(self, rules: Rules | None = None, seed: int | None = None) -> None:
        self.rules = rules or Rules()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.state = self._new_state()

    def _new_state(self) -> GameState:
        cards = deck(self.rules.num_decks, self.rules.num_jokers)
        self.rng.shuffle(cards)

        players = [PlayerState() for _ in range(self.rules.num_players)]
        for _ in range(self.rules.cards_per_player):
            for p in players:
                p.hand.append(cards.pop())

        discard = [cards.pop()]
        return GameState(
            players=players,
            current_player=0,
            stock_pile=cards,
            discard_pile=discard,
            melds_on_table=[],
            round_index=0,
            turn_number=0,
            finished=False,
            winner=None,
            phase="draw",
        )

    def reset(self) -> GameState:
        self.state = self._new_state()
        return self.state

    def current_hand(self) -> list[Card]:
        return self.state.players[self.state.current_player].hand

    def _draw_stock(self) -> None:
        if not self.state.stock_pile:
            top = self.state.discard_pile.pop()
            self.rng.shuffle(self.state.discard_pile)
            self.state.stock_pile = self.state.discard_pile
            self.state.discard_pile = [top]
        card = self.state.stock_pile.pop()
        self.current_hand().append(card)

    def _draw_discard(self) -> None:
        card = self.state.discard_pile.pop()
        self.current_hand().append(card)

    def _play_meld(self, meld_ix: int) -> bool:
        hand = self.current_hand()
        melds = find_all_melds(hand, self.rules, max_results=24)
        if meld_ix < 0 or meld_ix >= len(melds):
            return False

        meld = melds[meld_ix]
        for c in meld.cards:
            hand.remove(c)
        self.state.players[self.state.current_player].melds.append(meld.cards)
        self.state.melds_on_table.append(meld.cards)
        self.state.players[self.state.current_player].has_opened = True
        return True

    def _discard_from_index(self, hand_ix: int) -> bool:
        hand = self.current_hand()
        if hand_ix < 0 or hand_ix >= len(hand):
            return False
        self.state.discard_pile.append(hand.pop(hand_ix))
        return True

    def _end_turn(self) -> tuple[bool, float, dict]:
        player = self.state.players[self.state.current_player]
        info: dict = {}
        reward = 0.0

        if len(player.hand) == 0:
            self.state.finished = True
            self.state.winner = self.state.current_player
            reward = 100.0
            info["win"] = True
            return True, reward, info

        self.state.current_player = (self.state.current_player + 1) % self.rules.num_players
        self.state.phase = "draw"
        self.state.turn_number += 1

        if self.state.turn_number >= self.rules.max_round_turns:
            self.state.finished = True
            scores = [hand_points(p.hand) for p in self.state.players]
            winner = int(np.argmin(scores))
            self.state.winner = winner
            if winner == 0:
                reward = 50.0
                info["win"] = True
            else:
                reward = -float(scores[0])
                info["win"] = False
            info["forced_termination"] = True
            return True, reward, info

        return False, reward, info

    def step(self, action: int, is_agent_player: bool = True) -> StepResult:
        if self.state.finished:
            return StepResult(self.state, 0.0, True, {})

        valid = True
        reward = -0.01
        info: dict = {}

        if self.state.phase == "draw":
            if action == 0:
                self._draw_stock()
                self.state.phase = "meld"
            elif action == 1 and self.state.discard_pile:
                top = self.state.discard_pile[-1]
                hand = self.current_hand()
                if self.rules.must_meld_if_draw_discard:
                    candidates = discard_take_melds(hand, top, self.rules, max_results=24)
                    if not candidates:
                        valid = False
                    else:
                        self._draw_discard()
                        melds_after = find_all_melds(self.current_hand(), self.rules, max_results=24)
                        chosen = candidates[0]
                        meld_ix = next((i for i, m in enumerate(melds_after) if m.cards == chosen.cards), None)
                        if meld_ix is None or not self._play_meld(meld_ix):
                            picked = self.current_hand().pop()
                            self.state.discard_pile.append(picked)
                            valid = False
                        else:
                            reward += 2.0
                            if len(self.current_hand()) == 0:
                                done, terminal_reward, terminal_info = self._end_turn()
                                reward += terminal_reward
                                info.update(terminal_info)
                                info["win_after_meld"] = True
                                return StepResult(self.state, reward, done, info)
                            self.state.phase = "discard"
                            info["discard_auto_meld"] = True
                else:
                    self._draw_discard()
                    self.state.phase = "meld"
            else:
                valid = False

        elif self.state.phase == "meld":
            if action == 2:
                self.state.phase = "discard"
            elif action >= 3:
                hand_before = list(self.current_hand())
                meld_ix = action - 3
                valid = self._play_meld(meld_ix)
                if valid:
                    melds_before = find_all_melds(hand_before, self.rules, max_results=24)
                    chosen = melds_before[meld_ix]
                    jokers_in_meld = sum(1 for c in chosen.cards if c.is_joker)
                    natural_alternatives = 0
                    if jokers_in_meld > 0:
                        natural_alternatives = sum(
                            1
                            for m in melds_before
                            if len(m.cards) == len(chosen.cards) and all(not c.is_joker for c in m.cards)
                        )
                    reward += 2.0
                    info["meld_size"] = len(chosen.cards)
                    info["meld_kind"] = chosen.kind
                    info["meld_jokers_used"] = jokers_in_meld
                    info["meld_used_joker"] = jokers_in_meld > 0
                    info["meld_natural_alternatives"] = natural_alternatives
                    if len(self.current_hand()) == 0:
                        done, terminal_reward, terminal_info = self._end_turn()
                        reward += terminal_reward
                        info.update(terminal_info)
                        info["win_after_meld"] = True
                        return StepResult(self.state, reward, done, info)
                    # Allow chaining multiple melds in the same turn.
                    self.state.phase = "meld"
                else:
                    self.state.phase = "discard"
            else:
                valid = False
                self.state.phase = "discard"

        elif self.state.phase == "discard":
            discard_start = 3 + 24
            if action >= discard_start:
                discard_ix = action - discard_start
                hand = self.current_hand()
                if 0 <= discard_ix < len(hand):
                    info["discard_is_joker"] = bool(hand[discard_ix].is_joker)
                valid = self._discard_from_index(discard_ix)
            else:
                valid = False
            if valid:
                done, terminal_reward, terminal_info = self._end_turn()
                reward += terminal_reward
                info.update(terminal_info)
                return StepResult(self.state, reward, done, info)

        if not valid:
            reward -= 1.0
            info["invalid_action"] = True

        if is_agent_player:
            info["hand_points"] = hand_points(self.state.players[0].hand)

        return StepResult(self.state, reward, False, info)

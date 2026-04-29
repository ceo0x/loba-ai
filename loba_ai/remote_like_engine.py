from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from loba_ai.cards import Card, deck, hand_points
from loba_ai.melds import is_valid_run
from loba_ai.remote_like_actions import build_remote_like_legal_actions
from loba_ai.rules import Rules
from loba_ai.state import GameState, PlayerState


@dataclass(slots=True)
class RemoteLikeStepResult:
    state: GameState
    reward: float
    done: bool
    info: dict[str, Any]


class RemoteLikeGameEngine:
    def __init__(self, rules: Rules | None = None, seed: int | None = None) -> None:
        self.rules = rules or Rules()
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.table_melds: list[dict[str, Any]] = []
        self.next_meld_id = 0
        self.state = self._new_state()

    def _new_state(self) -> GameState:
        cards = deck(self.rules.num_decks, self.rules.num_jokers)
        self.rng.shuffle(cards)
        players = [PlayerState() for _ in range(self.rules.num_players)]
        for _ in range(self.rules.cards_per_player):
            for p in players:
                p.hand.append(cards.pop())
        discard = [cards.pop()]
        self.table_melds = []
        self.next_meld_id = 0
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

    def legal_actions(self) -> list[dict[str, Any]]:
        return build_remote_like_legal_actions(self.state, self.rules, self.table_melds)

    def _end_turn(self) -> tuple[bool, float, dict[str, Any]]:
        player = self.state.players[self.state.current_player]
        reward = 0.0
        info: dict[str, Any] = {}
        if len(player.hand) == 0:
            self.state.finished = True
            self.state.winner = self.state.current_player
            info["win"] = True
            reward = 100.0
            return True, reward, info
        self.state.current_player = (self.state.current_player + 1) % self.rules.num_players
        self.state.turn_number += 1
        self.state.phase = "draw"
        if self.state.turn_number >= self.rules.max_round_turns:
            self.state.finished = True
            scores = [hand_points(p.hand) for p in self.state.players]
            winner = int(np.argmin(scores))
            self.state.winner = winner
            info["forced_termination"] = True
            reward = 50.0 if winner == 0 else -float(scores[0])
            return True, reward, info
        return False, reward, info

    def _draw_stock(self) -> None:
        if not self.state.stock_pile:
            if len(self.state.discard_pile) > 1:
                top = self.state.discard_pile.pop()
                self.rng.shuffle(self.state.discard_pile)
                self.state.stock_pile = self.state.discard_pile
                self.state.discard_pile = [top]
        if self.state.stock_pile:
            self.state.players[self.state.current_player].hand.append(self.state.stock_pile.pop())

    def _remove_card_from_hand(self, card_payload: dict[str, Any]) -> Card | None:
        hand = self.state.players[self.state.current_player].hand
        hand_index = card_payload.get("hand_index")
        if isinstance(hand_index, int) and 0 <= hand_index < len(hand):
            candidate = hand[hand_index]
            if self._payload_matches_card(card_payload, candidate):
                return hand.pop(hand_index)
            # hand_index points to the wrong card (state desync) — fall back to structural match.
        # fallback structural match (rank + suit + deck_id, no order assumption)
        for i, c in enumerate(hand):
            if self._payload_matches_card(card_payload, c):
                return hand.pop(i)
        return None

    @staticmethod
    def _payload_matches_card(card_payload: dict[str, Any], c: Card) -> bool:
        if bool(card_payload.get("joker")) != bool(c.is_joker):
            return False
        if c.is_joker:
            return int(card_payload.get("deck_id", -1)) == c.deck_id
        # rank: payload uses "A"/"J"/"Q"/"K" or numeric strings; Card uses ints 1..13
        rank_long = c.rank if c.rank <= 10 else {11: "J", 12: "Q", 13: "K"}.get(c.rank, c.rank)
        rank_alt = {1: "A", 11: "J", 12: "Q", 13: "K"}.get(c.rank, c.rank)
        payload_rank = str(card_payload.get("rank"))
        if payload_rank != str(rank_long) and payload_rank != str(rank_alt):
            return False
        if int(card_payload.get("deck_id", -2)) != int(c.deck_id):
            return False
        # suit: payload uses short codes "C"/"D"/"H"/"S"; Card uses "clubs"/"diamonds"/"hearts"/"spades"
        suit_short_to_long = {"C": "clubs", "D": "diamonds", "H": "hearts", "S": "spades"}
        payload_suit = card_payload.get("suit")
        if payload_suit is None:
            # Some legal-action payloads omit suit; accept any matching rank+deck.
            return True
        payload_suit_long = suit_short_to_long.get(str(payload_suit), str(payload_suit))
        return payload_suit_long == str(c.suit)

    def _lay_meld(self, action: dict[str, Any]) -> bool:
        payload_cards = action.get("cards", [])
        if not isinstance(payload_cards, list) or not payload_cards:
            return False
        picked: list[Card] = []
        for cp in payload_cards:
            c = self._remove_card_from_hand(cp)
            if c is None:
                # rollback
                self.state.players[self.state.current_player].hand.extend(picked)
                return False
            picked.append(c)
        kind = "pierna" if action.get("type") == "LayPierna" else "escalera"
        self.table_melds.append(
            {"meld_id": self.next_meld_id, "owner": self.state.current_player, "kind": kind, "cards": list(picked)}
        )
        self.next_meld_id += 1
        self.state.players[self.state.current_player].has_opened = True
        self.state.players[self.state.current_player].melds.append(tuple(picked))
        self.state.melds_on_table.append(tuple(picked))
        return True

    def _extend_meld(self, action: dict[str, Any]) -> bool:
        meld_id = action.get("meld_id")
        card_payload = action.get("card")
        target = next((m for m in self.table_melds if m["meld_id"] == meld_id), None)
        if target is None or not isinstance(card_payload, dict):
            return False
        card = self._remove_card_from_hand(card_payload)
        if card is None:
            return False
        kind = target.get("kind")
        if kind == "escalera":
            trial = list(target["cards"]) + [card]
            if not is_valid_run(trial, self.rules):
                self.state.players[self.state.current_player].hand.append(card)
                return False
            target["cards"].append(card)
            return True
        if kind == "pierna":
            if card.is_joker:
                self.state.players[self.state.current_player].hand.append(card)
                return False
            naturals = [c for c in target["cards"] if not c.is_joker]
            if not naturals:
                self.state.players[self.state.current_player].hand.append(card)
                return False
            rank = naturals[0].rank
            used_suits = {c.suit for c in naturals}
            if card.rank != rank or card.suit not in used_suits:
                self.state.players[self.state.current_player].hand.append(card)
                return False
            target["cards"].append(card)
            return True
        self.state.players[self.state.current_player].hand.append(card)
        return False

    def _move_joker(self, action: dict[str, Any]) -> bool:
        """Strict MoveJoker:

        - Replacement must match the EXACT rank the end-joker represents AND the meld's suit.
        - The joker is moved to the opposite end; its new rank must be in [1, 13].
        - If both ends have jokers, the side is inferred from the replacement's rank.
        """
        meld_id = action.get("meld_id")
        replacement_payload = action.get("replacement")
        target = next(
            (m for m in self.table_melds if m["meld_id"] == meld_id and m["kind"] == "escalera"),
            None,
        )
        if target is None or not isinstance(replacement_payload, dict):
            return False
        if bool(replacement_payload.get("joker")):
            return False  # cannot replace with a joker

        cards: list[Card] = target["cards"]
        if len(cards) < 2:
            return False

        left_joker = cards[0].is_joker
        right_joker = cards[-1].is_joker
        if not left_joker and not right_joker:
            return False

        meld_suit = next((c.suit for c in cards if not c.is_joker), None)
        if meld_suit is None:
            return False
        first_natural = next((c for c in cards if not c.is_joker), None)
        last_natural = next((c for c in reversed(cards) if not c.is_joker), None)
        if first_natural is None or last_natural is None:
            return False

        # Determine intended side from the replacement payload.
        payload_rank_int = self._payload_rank_int(replacement_payload)
        payload_suit_long = self._payload_suit_long(replacement_payload)
        if payload_rank_int is None:
            return False
        if payload_suit_long is not None and payload_suit_long != meld_suit:
            return False

        side: str | None = None
        # LEFT side: replacement rank == first_natural.rank - 1 ; joker shifts right (must be <= 13).
        if left_joker:
            left_required_rank = first_natural.rank - 1
            new_right_rank = last_natural.rank + 1
            if (
                payload_rank_int == left_required_rank
                and 1 <= left_required_rank <= 13
                and 1 <= new_right_rank <= 13
            ):
                side = "left"
        # RIGHT side: replacement rank == last_natural.rank + 1 ; joker shifts left (must be >= 1).
        if side is None and right_joker:
            right_required_rank = last_natural.rank + 1
            new_left_rank = first_natural.rank - 1
            if (
                payload_rank_int == right_required_rank
                and 1 <= right_required_rank <= 13
                and 1 <= new_left_rank <= 13
            ):
                side = "right"
        if side is None:
            return False

        replacement = self._remove_card_from_hand(replacement_payload)
        if replacement is None:
            return False
        if replacement.suit != meld_suit or replacement.is_joker:
            # Defensive: structural matcher returned a wrong card; restore.
            self.state.players[self.state.current_player].hand.append(replacement)
            return False

        original_cards = list(cards)
        if side == "left":
            joker = cards.pop(0)
            cards.insert(0, replacement)
            cards.append(joker)
        else:
            joker = cards.pop(-1)
            cards.append(replacement)
            cards.insert(0, joker)

        # Defensive: should always pass given pre-checks, but rollback if not.
        if not is_valid_run(cards, self.rules):
            target["cards"] = original_cards
            self.state.players[self.state.current_player].hand.append(replacement)
            return False
        return True

    @staticmethod
    def _payload_rank_int(card_payload: dict[str, Any]) -> int | None:
        if bool(card_payload.get("joker")):
            return None
        raw = str(card_payload.get("rank"))
        rank_map = {"A": 1, "J": 11, "Q": 12, "K": 13}
        if raw in rank_map:
            return rank_map[raw]
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _payload_suit_long(card_payload: dict[str, Any]) -> str | None:
        suit = card_payload.get("suit")
        if suit is None:
            return None
        suit_short_to_long = {"C": "clubs", "D": "diamonds", "H": "hearts", "S": "spades"}
        return suit_short_to_long.get(str(suit), str(suit))

    def _discard(self, action: dict[str, Any]) -> bool:
        card_payload = action.get("card")
        if not isinstance(card_payload, dict):
            return False
        card = self._remove_card_from_hand(card_payload)
        if card is None:
            return False
        self.state.discard_pile.append(card)
        return True

    def step(self, action: dict[str, Any], is_agent_player: bool = True) -> RemoteLikeStepResult:
        if self.state.finished:
            return RemoteLikeStepResult(self.state, 0.0, True, {})
        valid = True
        reward = -0.01
        info: dict[str, Any] = {}
        action_type = action.get("type")
        phase = self.state.phase
        if phase == "draw":
            if action_type == "DrawStock":
                self._draw_stock()
                self.state.phase = "play_or_discard"
            elif action_type == "DrawDiscard" and self.state.discard_pile:
                self.state.players[self.state.current_player].hand.append(self.state.discard_pile.pop())
                play = action.get("play", {})
                if not isinstance(play, dict):
                    valid = False
                else:
                    if play.get("type") in {"LayPierna", "LayEscalera"}:
                        valid = self._lay_meld(play)
                    elif play.get("type") == "ExtendMeld":
                        valid = self._extend_meld(play)
                    elif play.get("type") == "MoveJoker":
                        valid = self._move_joker(play)
                    elif play.get("type") == "Cruzar":
                        valid = self._discard(play)
                    else:
                        valid = False
                self.state.phase = "play_or_discard"
                # "Loba seca" via DrawDiscard + meld play that empties the hand.
                if valid:
                    player = self.state.players[self.state.current_player]
                    if len(player.hand) == 0:
                        self.state.finished = True
                        self.state.winner = self.state.current_player
                        info["win"] = True
                        info["loba_seca"] = True
                        return RemoteLikeStepResult(self.state, reward + 100.0, True, info)
            else:
                valid = False
        elif phase == "play_or_discard":
            if action_type in {"LayPierna", "LayEscalera"}:
                valid = self._lay_meld(action)
            elif action_type == "ExtendMeld":
                valid = self._extend_meld(action)
            elif action_type == "MoveJoker":
                valid = self._move_joker(action)
            elif action_type == "Cruzar":
                valid = self._discard(action)
                if valid:
                    # Stay in play_or_discard. Player can chain more cruces, melds,
                    # extends, or eventually a final Discard to end the turn.
                    return RemoteLikeStepResult(self.state, reward, False, {"cruzar": True})
            elif action_type == "Discard":
                valid = self._discard(action)
                if valid:
                    done, end_reward, end_info = self._end_turn()
                    return RemoteLikeStepResult(self.state, reward + end_reward, done, end_info)
            else:
                valid = False

            # "Loba seca": going out by emptying the hand via meld/extend/joker without discarding.
            # The engine awards the same +100 win bonus as a regular Discard-loba.
            if valid and action_type in {"LayPierna", "LayEscalera", "ExtendMeld", "MoveJoker"}:
                player = self.state.players[self.state.current_player]
                if len(player.hand) == 0:
                    self.state.finished = True
                    self.state.winner = self.state.current_player
                    info["win"] = True
                    info["loba_seca"] = True
                    return RemoteLikeStepResult(self.state, reward + 100.0, True, info)
        if not valid:
            reward -= 1.0
            info["invalid_action"] = True
        if is_agent_player:
            info["hand_points"] = hand_points(self.state.players[0].hand)
        return RemoteLikeStepResult(self.state, reward, False, info)

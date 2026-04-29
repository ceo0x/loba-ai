from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np

from loba_ai.agents.heuristic_agent import HeuristicAgent
from loba_ai.agents.random_agent import RandomAgent
from loba_ai.env import LobaEnv, MAX_MELD_ACTIONS
from loba_ai.hand_display import build_hand_display, card_label
from loba_ai.melds import find_all_melds
from loba_ai.model_io import choose_action, load_model

PlayerKind = Literal["human", "heuristic", "random", "model"]


@dataclass(slots=True)
class PlayerConfig:
    kind: PlayerKind
    model_name: str | None = None


@dataclass(slots=True)
class RegisteredModel:
    name: str
    path: str
    model: object


class MatchSession:
    def __init__(
        self,
        env: LobaEnv,
        players: list[PlayerConfig],
        model_registry: dict[str, RegisteredModel],
        seed: int | None = None,
        autoplay_enabled: bool = False,
    ):
        self.match_id = uuid4().hex
        self.env = env
        self.players = players
        self.model_registry = model_registry
        self.autoplay_enabled = autoplay_enabled
        self.rng = np.random.default_rng(seed)
        self.random_agent = RandomAgent(self.rng)
        self.heuristic_agent = HeuristicAgent(self.rng)
        self.last_reward = 0.0
        self.obs, self.info = self.env.reset(seed=seed)

    @property
    def state(self):
        return self.env.engine.state

    def _card_label(self, c) -> str:
        return card_label(c)

    def _card_sort_key(self, c) -> tuple[int, int]:
        if c.is_joker:
            return (99, 99)
        suit_order = {"clubs": 0, "diamonds": 1, "hearts": 2, "spades": 3}
        return (c.rank, suit_order.get(c.suit or "clubs", 9))

    def _sorted_card_labels(self, cards) -> list[str]:
        sorted_cards = sorted(cards, key=self._card_sort_key)
        return [self._card_label(c) for c in sorted_cards]

    def _valid_actions(self) -> list[int]:
        mask = self.info["action_mask"]
        return [i for i, v in enumerate(mask.tolist()) if v]

    def _action_label(self, action: int) -> str:
        if action == 0:
            return "Draw stock"
        if action == 1:
            return "Draw discard"
        if action == 2:
            return "Skip meld"

        meld_start = 3
        meld_end = meld_start + MAX_MELD_ACTIONS
        if meld_start <= action < meld_end:
            meld_ix = action - meld_start
            melds = find_all_melds(self.state.players[self.state.current_player].hand, self.env.rules, max_results=MAX_MELD_ACTIONS)
            if meld_ix < len(melds):
                cards = "-".join(self._card_label(c) for c in melds[meld_ix].cards)
                return f"Play meld {cards}"
            return f"Play meld #{meld_ix}"

        discard_start = meld_end
        hand_ix = action - discard_start
        hand = self.state.players[self.state.current_player].hand
        if 0 <= hand_ix < len(hand):
            return f"Discard {self._card_label(hand[hand_ix])}"
        return f"Action {action}"

    def _sorted_valid_actions_payload(self, valid_actions: list[int]) -> list[dict]:
        s = self.state
        payload = [{"id": a, "label": self._action_label(a)} for a in valid_actions]
        if s.phase != "meld":
            return payload

        meld_start = 3
        meld_end = meld_start + MAX_MELD_ACTIONS
        prefix = [p for p in payload if p["id"] < meld_start]
        melds = [p for p in payload if meld_start <= p["id"] < meld_end]
        suffix = [p for p in payload if p["id"] >= meld_end]

        def meld_sort_key(item: dict) -> tuple[int, str]:
            # Sort visual list by readable meld label while keeping real action ids.
            return (len(item["label"]), item["label"])

        melds.sort(key=meld_sort_key)
        return prefix + melds + suffix

    def _action_event(self, player_ix: int, action: int) -> dict:
        phase = self.state.phase
        event: dict = {"player": player_ix, "action": action, "label": self._action_label(action), "phase": phase}

        if phase == "draw":
            if action == 0 and self.state.stock_pile:
                event["drawn_card"] = self._card_label(self.state.stock_pile[-1])
                event["draw_source"] = "stock"
            elif action == 1 and self.state.discard_pile:
                event["drawn_card"] = self._card_label(self.state.discard_pile[-1])
                event["draw_source"] = "discard"

        if phase == "meld":
            meld_start = 3
            meld_end = meld_start + MAX_MELD_ACTIONS
            if meld_start <= action < meld_end:
                meld_ix = action - meld_start
                melds = find_all_melds(self.state.players[player_ix].hand, self.env.rules, max_results=MAX_MELD_ACTIONS)
                if meld_ix < len(melds):
                    event["meld_cards"] = [self._card_label(c) for c in melds[meld_ix].cards]

        if phase == "discard":
            discard_start = 3 + MAX_MELD_ACTIONS
            hand_ix = action - discard_start
            hand = self.state.players[player_ix].hand
            if 0 <= hand_ix < len(hand):
                event["discarded_card"] = self._card_label(hand[hand_ix])

        return event

    def _to_public_state(self, viewer_player_index: int | None = None) -> dict:
        s = self.state
        me = s.current_player
        top_discard = self._card_label(s.discard_pile[-1]) if s.discard_pile else None
        discard_last_three = [self._card_label(c) for c in s.discard_pile[-3:]] if s.discard_pile else []

        players_payload = []
        table_melds_grouped = []
        for idx, p in enumerate(s.players):
            owner_melds = [self._sorted_card_labels(meld) for meld in p.melds]
            hand_display = build_hand_display(p.hand, self.env.rules)
            can_view_hand = (
                viewer_player_index is None
                and (idx == me or self.players[idx].kind == "human")
            ) or (viewer_player_index is not None and idx == viewer_player_index)
            players_payload.append(
                {
                    "index": idx,
                    "kind": self.players[idx].kind,
                    "model_name": self.players[idx].model_name,
                    "cards_in_hand": len(p.hand),
                    "has_opened": p.has_opened,
                    "hand": [self._card_label(c) for c in p.hand] if can_view_hand else [],
                    "hand_full": [self._card_label(c) for c in p.hand] if can_view_hand else [],
                    "hand_display": hand_display if can_view_hand else [],
                    "melds": owner_melds,
                }
            )
            table_melds_grouped.append(
                {
                    "player_index": idx,
                    "kind": self.players[idx].kind,
                    "model_name": self.players[idx].model_name,
                    "melds": owner_melds,
                }
            )

        valid_actions = self._valid_actions()
        discard_start = 3 + MAX_MELD_ACTIONS
        discard_options = []
        is_viewer_turn = viewer_player_index is None or viewer_player_index == s.current_player
        if s.phase == "discard" and is_viewer_turn:
            hand = s.players[s.current_player].hand
            for a in valid_actions:
                if a >= discard_start:
                    hand_ix = a - discard_start
                    if 0 <= hand_ix < len(hand):
                        discard_options.append(
                            {
                                "action_id": a,
                                "hand_index": hand_ix,
                                "card": self._card_label(hand[hand_ix]),
                            }
                        )

        return {
            "match_id": self.match_id,
            "phase": s.phase,
            "turn": s.turn_number,
            "current_player": me,
            "autoplay_enabled": self.autoplay_enabled,
            "finished": s.finished,
            "winner": s.winner,
            "winner_detail": (
                {
                    "index": s.winner,
                    "kind": self.players[s.winner].kind,
                    "model_name": self.players[s.winner].model_name,
                }
                if s.winner is not None
                else None
            ),
            "stock_size": len(s.stock_pile),
            "top_discard": top_discard,
            "discard_last_three": discard_last_three,
            "table_melds": [self._sorted_card_labels(meld) for meld in s.melds_on_table],
            "table_melds_grouped": table_melds_grouped,
            "players": players_payload,
            "valid_actions": self._sorted_valid_actions_payload(valid_actions) if is_viewer_turn else [],
            "discard_options": discard_options,
            "last_reward": self.last_reward,
        }

    def snapshot(self) -> dict:
        return self._to_public_state()

    def snapshot_for_player(self, player_index: int) -> dict:
        if player_index < 0 or player_index >= len(self.players):
            raise ValueError("Invalid player index")
        return self._to_public_state(viewer_player_index=player_index)

    def step(self, action: int) -> dict:
        actor = self.state.current_player
        event = self._action_event(actor, action)
        obs, reward, done, _, info = self.env.step(action)
        self.obs, self.info = obs, info
        self.last_reward = float(reward)
        return {
            "done": done,
            "event": event,
            "state": self._to_public_state(),
        }

    def _choose_bot_action(self, player_ix: int) -> int:
        cfg = self.players[player_ix]
        mask = self.info["action_mask"]
        phase = self.state.phase
        if cfg.kind == "human":
            if self.autoplay_enabled:
                return self.heuristic_agent.act(mask, phase)
            raise ValueError("Human player has no automatic action")
        if cfg.kind == "random":
            return self.random_agent.act(mask)
        if cfg.kind == "heuristic":
            return self.heuristic_agent.act(mask, phase)
        if cfg.kind == "model":
            if not cfg.model_name or cfg.model_name not in self.model_registry:
                raise ValueError("Model player configured without registered model")
            model = self.model_registry[cfg.model_name].model
            return choose_action(model, self.obs, mask)
        raise ValueError("Unsupported player kind")

    def autoplay_until_human_or_end(self, max_steps: int = 5000) -> dict:
        log: list[dict] = []
        for _ in range(max_steps):
            if self.state.finished:
                break
            current = self.state.current_player
            cfg = self.players[current]
            if cfg.kind == "human" and not self.autoplay_enabled:
                break
            action = self._choose_bot_action(current)
            result = self.step(action)
            entry = dict(result["event"])
            entry["done"] = result["done"]
            log.append(entry)
            if result["done"]:
                break

        return {"log": log, "state": self._to_public_state()}


class PlaygroundService:
    def __init__(self) -> None:
        self.matches: dict[str, MatchSession] = {}
        self.models: dict[str, RegisteredModel] = {}

    def register_model(self, name: str, path: str) -> dict:
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        model = load_model(str(model_path))
        reg = RegisteredModel(name=name, path=str(model_path), model=model)
        self.models[name] = reg
        return {"name": reg.name, "path": reg.path}

    def list_models(self) -> list[dict]:
        return [{"name": m.name, "path": m.path} for m in self.models.values()]

    def remove_model(self, name: str) -> None:
        if name not in self.models:
            raise KeyError(f"Model '{name}' not found")
        del self.models[name]

    def create_match(self, players: list[PlayerConfig], seed: int | None = None, autoplay_enabled: bool = False) -> dict:
        if len(players) < 2 or len(players) > 6:
            raise ValueError("Playground supports between 2 and 6 players")

        for p in players:
            if p.kind == "model" and (not p.model_name or p.model_name not in self.models):
                raise ValueError(f"Unknown model '{p.model_name}'")

        env = LobaEnv(opponent="manual", seed=seed)
        env.engine.rules.num_players = len(players)
        match = MatchSession(
            env=env,
            players=players,
            model_registry=self.models,
            seed=seed,
            autoplay_enabled=autoplay_enabled,
        )
        self.matches[match.match_id] = match

        auto = match.autoplay_until_human_or_end()
        return {"match_id": match.match_id, "state": auto["state"], "autoplay_log": auto["log"]}

    def get_match(self, match_id: str) -> MatchSession:
        if match_id not in self.matches:
            raise KeyError("Match not found")
        return self.matches[match_id]

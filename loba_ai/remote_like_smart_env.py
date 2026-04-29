from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from loba_ai.agents.random_agent import RandomAgent
from loba_ai.remote_like_actions import canonicalize_remote_like_legal_actions
from loba_ai.remote_like_engine import RemoteLikeGameEngine
from loba_ai.remote_like_obs_builder import RUN_PROJECT_MAX_RANK_GAP
from loba_ai.rules import Rules

MAX_REMOTE_LIKE_ACTIONS = 256

_SUIT_TO_OFFSET = {"C": 0, "D": 13, "H": 26, "S": 39}
_OFFSET_TO_SUIT = {v: k for k, v in _SUIT_TO_OFFSET.items()}


class RemoteLikeSmartLobaEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        rules: Rules | None = None,
        seed: int | None = None,
        opponent: str = "random",
        discard_history_window: int = 2,
    ) -> None:
        super().__init__()
        self.rules = rules or Rules()
        self.engine = RemoteLikeGameEngine(self.rules, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.random_agent = RandomAgent(self.rng)
        self.opponent_type = opponent
        self.discard_history_window = max(1, int(discard_history_window))
        self._last_legal_actions: list[dict] = []

        # Shaping coefficients (smart v1 media).
        self.reward_play_action_bonus = 0.7
        self.reward_extend_action_bonus = 0.5
        self.reward_cruzar_action_bonus = 0.4
        self.reward_discard_with_play_options_penalty = 1.1
        self.reward_discard_project_penalty = 0.5
        self.reward_discard_low_single_before_high_single_penalty = 0.2
        self.reward_discard_hot_card_penalty = 0.25

        self._seen_token_counts = np.zeros(53, dtype=np.float32)
        # Set of unique card identifiers we've seen publicly. Prevents the legacy
        # bug where each step re-counted the agent's hand and saturated the seen
        # counts at 4 within a couple of turns.
        self._seen_unique_ids: set[tuple] = set()
        self._recent_discards_by_player: list[list[int]] = [[] for _ in range(self.rules.num_players)]
        self._episode_discard_with_play_options = 0
        self._episode_play_when_available = 0
        self._episode_play_options_turns = 0

        # MoveJoker loop protection: count consecutive MoveJoker actions in the current turn.
        # When >= max_consecutive_move_jokers, suppress MoveJoker from legal actions until
        # the actor changes or a non-MoveJoker action is taken.
        self._consecutive_move_jokers = 0
        self._last_actor_for_move_joker_guard: int | None = None
        self.max_consecutive_move_jokers = 2

        base_dim = 111 + (2 * int(self.rules.num_players))
        seen_dim = 53
        recent_discard_dim = 53 * self.discard_history_window * max(0, int(self.rules.num_players) - 1)
        hand_feature_dim = 3  # pair_count, near_run_count, dead_card_count
        obs_dim = base_dim + seen_dim + recent_discard_dim + hand_feature_dim

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(MAX_REMOTE_LIKE_ACTIONS)

    @staticmethod
    def _card_payload(card) -> dict:
        if card.is_joker:
            return {"joker": True, "deck_id": card.deck_id}
        rank_map = {1: "A", 11: "J", 12: "Q", 13: "K"}
        suit_map = {"spades": "S", "hearts": "H", "clubs": "C", "diamonds": "D"}
        return {"rank": rank_map.get(card.rank, str(card.rank)), "suit": suit_map.get(card.suit or "clubs", "C"), "deck_id": card.deck_id}

    @staticmethod
    def _card_token(card_payload: dict) -> int:
        if card_payload.get("joker"):
            return 52
        suit = str(card_payload.get("suit", ""))
        rank_raw = str(card_payload.get("rank", "0"))
        rank_map = {"A": 1, "J": 11, "Q": 12, "K": 13}
        if suit not in _SUIT_TO_OFFSET:
            return 52
        try:
            rank_num = int(rank_raw)
        except ValueError:
            rank_num = rank_map.get(rank_raw, 0)
        if rank_num <= 0:
            return 52
        return _SUIT_TO_OFFSET[suit] + max(0, min(12, rank_num - 1))

    @staticmethod
    def _token_rank_suit(token: int) -> tuple[int, str]:
        if token >= 52:
            return (99, "J")
        offset = (token // 13) * 13
        rank = (token % 13) + 1
        suit = _OFFSET_TO_SUIT.get(offset, "C")
        return rank, suit

    @staticmethod
    def _is_payload_joker(card_payload: dict) -> bool:
        return bool(card_payload.get("joker"))

    @staticmethod
    def _payload_key(card_payload: dict) -> str:
        if bool(card_payload.get("joker")):
            return f"J|{card_payload.get('deck_id')}"
        return f"{card_payload.get('rank')}|{card_payload.get('suit')}|{card_payload.get('deck_id')}"

    @staticmethod
    def _payload_rank_value(card_payload: dict) -> int:
        if bool(card_payload.get("joker")):
            return 99
        raw = str(card_payload.get("rank", "0"))
        rank_map = {"A": 14, "J": 11, "Q": 12, "K": 13}
        if raw in rank_map:
            return rank_map[raw]
        try:
            return int(raw)
        except ValueError:
            return 0

    @staticmethod
    def _payload_run_rank_value(card_payload: dict) -> int:
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

    @staticmethod
    def _payload_suit(card_payload: dict) -> str:
        return str(card_payload.get("suit", ""))

    def _is_project_card_payload(self, card_payload: dict, hand_payload: list[dict]) -> bool:
        if self._is_payload_joker(card_payload):
            return True
        rank_v = self._payload_rank_value(card_payload)
        suit = self._payload_suit(card_payload)
        if rank_v <= 0:
            return False
        run_rank_v = self._payload_run_rank_value(card_payload)
        target_key = self._payload_key(card_payload)
        same_rank = 0
        same_suit_neighbors = 0
        for c in hand_payload:
            if self._payload_key(c) == target_key:
                continue
            if self._is_payload_joker(c):
                continue
            rv = self._payload_rank_value(c)
            if rv == rank_v and self._payload_suit(c) != suit:
                same_rank += 1
            rank_gap = abs(self._payload_run_rank_value(c) - run_rank_v)
            if self._payload_suit(c) == suit and 1 <= rank_gap <= RUN_PROJECT_MAX_RANK_GAP:
                same_suit_neighbors += 1
        return same_rank >= 1 or same_suit_neighbors >= 1

    def _singleton_non_project_rank_values(self, hand_payload: list[dict]) -> list[int]:
        rank_counts: dict[int, int] = {}
        for c in hand_payload:
            if self._is_payload_joker(c):
                continue
            rv = self._payload_rank_value(c)
            if rv <= 0:
                continue
            rank_counts[rv] = rank_counts.get(rv, 0) + 1
        out: list[int] = []
        for c in hand_payload:
            if self._is_payload_joker(c):
                continue
            rv = self._payload_rank_value(c)
            if rv > 0 and rank_counts.get(rv, 0) == 1 and not self._is_project_card_payload(c, hand_payload):
                out.append(rv)
        return out

    def _is_hot_discard_token(self, token: int, actor_index: int) -> bool:
        if token >= 52:
            return False
        rank, suit = self._token_rank_suit(token)
        for rival_idx, hist in enumerate(self._recent_discards_by_player):
            if rival_idx == actor_index:
                continue
            for seen in hist:
                if seen >= 52:
                    continue
                s_rank, s_suit = self._token_rank_suit(seen)
                if s_rank == rank:
                    return True
                if s_suit == suit and abs(s_rank - rank) <= 1:
                    return True
        return False

    def _build_observation_payload(self, seat: int | None = None) -> dict:
        s = self.engine.state
        if seat is None:
            seat = int(s.current_player)
        return {
            "seat": seat,
            "num_players": self.rules.num_players,
            "phase": s.phase,
            "hand": [self._card_payload(c) for c in s.players[seat].hand],
            "other_hand_sizes": [len(p.hand) for p in s.players],
            "stock_size": len(s.stock_pile),
            "discard_top": self._card_payload(s.discard_pile[-1]) if s.discard_pile else None,
            "discard_size": len(s.discard_pile),
            "pending_discard": None,
            "melds_on_table": [
                {
                    "meld_id": m["meld_id"],
                    "owner": m["owner"],
                    "kind": m["kind"],
                    "cards": [self._card_payload(c) for c in m["cards"]],
                }
                for m in self.engine.table_melds
            ],
            "has_laid_meld_this_round": [bool(p.has_opened) for p in s.players],
            "cumulative_scores": [int(p.score) for p in s.players],
            "reenganches_used": [0 for _ in s.players],
            "eliminated": [False for _ in s.players],
            "legal_actions": list(self._last_legal_actions),
        }

    @staticmethod
    def _phase_vector(phase: str) -> np.ndarray:
        vec = np.zeros(3, dtype=np.float32)
        mapping = {"draw": 0, "play_or_discard": 1, "cruzar_discard": 2}
        vec[mapping.get(phase, 0)] = 1.0
        return vec

    def _base_obs_vector(self, payload: dict) -> np.ndarray:
        hand = np.zeros(53, dtype=np.float32)
        for card in payload.get("hand", []):
            hand[self._card_token(card)] += 1.0
        hand /= 4.0

        top_discard = np.zeros(53, dtype=np.float32)
        discard = payload.get("discard_top")
        if isinstance(discard, dict):
            top_discard[self._card_token(discard)] += 1.0

        my_seat = int(payload.get("seat", 0))
        num_players = max(2, int(payload.get("num_players", 2)))
        other_sizes = payload.get("other_hand_sizes", [])
        laid = payload.get("has_laid_meld_this_round", [])
        seat_order = list(range(my_seat, num_players)) + list(range(0, my_seat))
        counts_values = []
        opened_values = []
        for seat in seat_order:
            size = float(other_sizes[seat]) if seat < len(other_sizes) else 0.0
            counts_values.append(size / 10.0)
            opened_values.append(1.0 if (seat < len(laid) and laid[seat]) else 0.0)

        stock = float(payload.get("stock_size", 0)) / 108.0
        phase = self._phase_vector(str(payload.get("phase", "draw")))
        pad = np.zeros(1, dtype=np.float32)
        return np.concatenate(
            [
                hand,
                top_discard,
                np.array(counts_values, dtype=np.float32),
                np.array(opened_values, dtype=np.float32),
                np.array([stock], dtype=np.float32),
                phase,
                pad,
            ]
        ).astype(np.float32)

    def _hand_project_features(self, hand_payload: list[dict]) -> np.ndarray:
        pair_count = 0
        near_run_count = 0
        dead_card_count = 0

        rank_counts: dict[int, int] = {}
        suit_to_ranks: dict[str, list[int]] = {}
        for c in hand_payload:
            if self._is_payload_joker(c):
                continue
            rv = self._payload_rank_value(c)
            if rv <= 0:
                continue
            rank_counts[rv] = rank_counts.get(rv, 0) + 1
            suit_to_ranks.setdefault(self._payload_suit(c), []).append(rv)

        pair_count = sum(1 for v in rank_counts.values() if v >= 2)
        for ranks in suit_to_ranks.values():
            uniq = sorted(set(ranks))
            for idx in range(1, len(uniq)):
                if abs(uniq[idx] - uniq[idx - 1]) == 1:
                    near_run_count += 1
        for c in hand_payload:
            if not self._is_project_card_payload(c, hand_payload):
                dead_card_count += 1

        return np.array(
            [
                min(1.0, pair_count / 6.0),
                min(1.0, near_run_count / 8.0),
                min(1.0, dead_card_count / 14.0),
            ],
            dtype=np.float32,
        )

    def _recent_discards_vector(self, viewer_index: int) -> np.ndarray:
        # Slots rotated by viewer: first slot = viewer's clockwise neighbor.
        # Consistent with seat_order rotation in _base_obs_vector and match features.
        num_rivals = max(0, self.rules.num_players - 1)
        out = np.zeros(53 * self.discard_history_window * num_rivals, dtype=np.float32)
        cursor = 0
        for offset in range(1, self.rules.num_players):
            seat = (viewer_index + offset) % self.rules.num_players
            hist = self._recent_discards_by_player[seat][-self.discard_history_window :]
            # left-pad so the newest discards sit in stable slots regardless of count
            hist = ([-1] * (self.discard_history_window - len(hist))) + list(hist)
            for token in hist:
                if 0 <= token < 53:
                    out[cursor + token] = 1.0
                cursor += 53
        return out

    def _track_public_seen_cards(self) -> None:
        """Track first-time PUBLIC sightings. Does NOT include own hand (those are
        counted dynamically at obs time via the hand vector). Idempotent: safe to
        call repeatedly because we use a set of unique card identifiers.
        """
        s = self.engine.state
        # Fresh top of discard.
        if s.discard_pile:
            top_card = s.discard_pile[-1]
            uid = self._card_unique_id(top_card)
            if uid not in self._seen_unique_ids:
                self._seen_unique_ids.add(uid)
                self._seen_token_counts[self._card_token(self._card_payload(top_card))] += 1.0
        # All cards in melds on the table (cards become public when laid).
        for meld in self.engine.table_melds:
            for c in meld.get("cards", []) or []:
                uid = self._card_unique_id(c)
                if uid not in self._seen_unique_ids:
                    self._seen_unique_ids.add(uid)
                    self._seen_token_counts[self._card_token(self._card_payload(c))] += 1.0
        np.clip(self._seen_token_counts, 0.0, 4.0, out=self._seen_token_counts)

    @staticmethod
    def _card_unique_id(card_or_payload) -> tuple:
        """Stable unique identifier across Card objects and payload dicts.

        Layout: (rank_int, suit_long, deck_id, is_joker). Naturals get rank as int
        (A=1, J=11, Q=12, K=13). Suits are normalized to long form ('clubs' etc).
        """
        if isinstance(card_or_payload, dict):
            if bool(card_or_payload.get("joker")):
                return (0, None, int(card_or_payload.get("deck_id", -1)), True)
            raw = str(card_or_payload.get("rank", "0"))
            rank_map = {"A": 1, "J": 11, "Q": 12, "K": 13}
            if raw in rank_map:
                rank_int = rank_map[raw]
            else:
                try:
                    rank_int = int(raw)
                except ValueError:
                    rank_int = 0
            suit_short_to_long = {"C": "clubs", "D": "diamonds", "H": "hearts", "S": "spades"}
            suit_raw = str(card_or_payload.get("suit", ""))
            suit = suit_short_to_long.get(suit_raw, suit_raw)
            return (rank_int, suit, int(card_or_payload.get("deck_id", -1)), False)
        # Card object
        if bool(getattr(card_or_payload, "is_joker", False)):
            return (0, None, int(card_or_payload.deck_id), True)
        return (int(card_or_payload.rank), card_or_payload.suit, int(card_or_payload.deck_id), False)

    def _record_discard_event(self, actor_index: int, action: dict, valid: bool) -> None:
        if not valid:
            return
        action_type = str(action.get("type", ""))
        if action_type not in {"Discard", "Cruzar"}:
            return
        payload = action.get("card")
        if not isinstance(payload, dict):
            return
        token = self._card_token(payload)
        # Dedup via unique id so the same physical card isn't counted twice.
        uid = self._card_unique_id(payload)
        if uid not in self._seen_unique_ids:
            self._seen_unique_ids.add(uid)
            self._seen_token_counts[token] += 1.0
            np.clip(self._seen_token_counts, 0.0, 4.0, out=self._seen_token_counts)
        self._recent_discards_by_player[actor_index].append(token)
        self._recent_discards_by_player[actor_index] = self._recent_discards_by_player[actor_index][-self.discard_history_window :]

    def _build_obs_for_seat(self, seat: int) -> np.ndarray:
        payload = self._build_observation_payload(seat=seat)
        base = self._base_obs_vector(payload)
        seen = (self._seen_token_counts / 4.0).astype(np.float32)
        recent = self._recent_discards_vector(viewer_index=int(payload["seat"]))
        hand_features = self._hand_project_features(payload.get("hand", []))
        return np.concatenate([base, seen, recent, hand_features]).astype(np.float32)

    def _obs(self) -> np.ndarray:
        return self._build_obs_for_seat(seat=int(self.engine.state.current_player))

    def _refresh_legal_actions(self) -> None:
        actions = canonicalize_remote_like_legal_actions(self.engine.legal_actions())
        # Suppress MoveJoker if the current actor has already used it too many times this turn.
        current_actor = int(self.engine.state.current_player)
        if (
            self._last_actor_for_move_joker_guard == current_actor
            and self._consecutive_move_jokers >= self.max_consecutive_move_jokers
        ):
            filtered = [a for a in actions if str(a.get("type")) != "MoveJoker"]
            # Only apply the filter if we don't strip the actor's only options.
            if filtered:
                actions = filtered
        self._last_legal_actions = actions

    def _track_move_joker_guard(self, actor_index: int, action: dict) -> None:
        action_type = str(action.get("type", ""))
        if self._last_actor_for_move_joker_guard != actor_index:
            self._last_actor_for_move_joker_guard = actor_index
            self._consecutive_move_jokers = 0
        if action_type == "MoveJoker":
            self._consecutive_move_jokers += 1
        else:
            self._consecutive_move_jokers = 0

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        n = min(len(self._last_legal_actions), self.action_space.n)
        if n:
            mask[:n] = 1
        return mask

    def action_masks(self) -> np.ndarray:
        return self._action_mask().astype(bool)

    @staticmethod
    def _has_non_discard_play_options(legal_actions: list[dict]) -> bool:
        # MoveJoker excluded: avoiding a useless MoveJoker (which can loop and consume
        # a card with no real progress) by Discarding shouldn't be penalized as if the
        # agent ignored a real play opportunity.
        playish = {"LayPierna", "LayEscalera", "ExtendMeld", "Cruzar"}
        return any(str(a.get("type")) in playish for a in legal_actions)

    def _play_opponents(self) -> tuple[float, bool]:
        total_reward = 0.0
        while not self.engine.state.finished and self.engine.state.current_player != 0:
            self._refresh_legal_actions()
            if not self._last_legal_actions:
                return total_reward - 0.5, True
            opp_idx = int(self.random_agent.act(self._action_mask()))
            opp_idx = max(0, min(opp_idx, len(self._last_legal_actions) - 1))
            action = self._last_legal_actions[opp_idx]
            actor_index = int(self.engine.state.current_player)
            self._track_move_joker_guard(actor_index, action)
            res = self.engine.step(action, is_agent_player=False)
            self._record_discard_event(actor_index, action, not bool(res.info.get("invalid_action")))
            self._track_public_seen_cards()
            if res.done:
                if self.engine.state.winner == 0:
                    total_reward += 100.0
                else:
                    total_reward -= float(len(self.engine.state.players[0].hand))
                return total_reward, True
        return total_reward, False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.random_agent = RandomAgent(self.rng)
        self.engine.reset()
        self._seen_token_counts.fill(0.0)
        self._seen_unique_ids = set()
        self._recent_discards_by_player = [[] for _ in range(self.rules.num_players)]
        self._episode_discard_with_play_options = 0
        self._episode_play_when_available = 0
        self._episode_play_options_turns = 0
        self._track_public_seen_cards()
        self._refresh_legal_actions()
        obs = self._obs()
        if obs.shape[0] != self.observation_space.shape[0]:
            raise ValueError(
                f"RemoteLikeSmartLobaEnv observation shape mismatch: got {obs.shape[0]}, expected {self.observation_space.shape[0]}"
            )
        return obs, {"action_mask": self._action_mask(), "legal_actions": list(self._last_legal_actions)}

    def step(self, action: int):
        bootstrap_reward = 0.0
        if self.engine.state.current_player != 0:
            opp_reward, opp_done = self._play_opponents()
            bootstrap_reward += float(opp_reward)
            if opp_done:
                self._refresh_legal_actions()
                obs = self._obs()
                return obs, bootstrap_reward, True, False, {
                    "action_mask": self._action_mask(),
                    "legal_actions": list(self._last_legal_actions),
                    "autoplay_before_action": True,
                }

        self._refresh_legal_actions()
        pre_legal_actions = list(self._last_legal_actions)
        pre_phase = str(self.engine.state.phase)
        pre_hand_payload = [self._card_payload(c) for c in self.engine.state.players[0].hand]

        if not self._last_legal_actions:
            obs = self._obs()
            return obs, -1.0 + bootstrap_reward, True, False, {"action_mask": self._action_mask(), "invalid_action": True}

        action_ix = int(action)
        if action_ix < 0 or action_ix >= len(self._last_legal_actions):
            action_ix = 0
        selected = self._last_legal_actions[action_ix]
        actor_index = int(self.engine.state.current_player)
        self._track_move_joker_guard(actor_index, selected)
        res = self.engine.step(selected, is_agent_player=True)
        valid = not bool(res.info.get("invalid_action"))
        self._record_discard_event(actor_index, selected, valid)
        self._track_public_seen_cards()

        reward = float(res.reward) + bootstrap_reward
        selected_type = str(selected.get("type", "Unknown"))
        had_play_options = self._has_non_discard_play_options(pre_legal_actions)
        if pre_phase == "play_or_discard" and had_play_options:
            self._episode_play_options_turns += 1
            if selected_type in {"LayPierna", "LayEscalera", "ExtendMeld", "MoveJoker", "Cruzar"}:
                self._episode_play_when_available += 1
            elif selected_type == "Discard":
                self._episode_discard_with_play_options += 1

        if pre_phase == "play_or_discard":
            if selected_type in {"LayPierna", "LayEscalera"}:
                reward += self.reward_play_action_bonus
            elif selected_type in {"ExtendMeld", "MoveJoker"}:
                reward += self.reward_extend_action_bonus
            elif selected_type == "Cruzar":
                reward += self.reward_cruzar_action_bonus
            elif selected_type == "Discard" and had_play_options:
                reward -= self.reward_discard_with_play_options_penalty

            if selected_type == "Discard":
                selected_card = selected.get("card", {})
                if isinstance(selected_card, dict):
                    if self._is_project_card_payload(selected_card, pre_hand_payload):
                        reward -= self.reward_discard_project_penalty
                    singleton_values = self._singleton_non_project_rank_values(pre_hand_payload)
                    selected_rank = self._payload_rank_value(selected_card)
                    if singleton_values and selected_rank in singleton_values:
                        highest_single = max(singleton_values)
                        if selected_rank < highest_single:
                            reward -= self.reward_discard_low_single_before_high_single_penalty
                    selected_token = self._card_token(selected_card)
                    if self._is_hot_discard_token(selected_token, actor_index=0):
                        reward -= self.reward_discard_hot_card_penalty

        terminated = bool(res.done)
        if not terminated and self.engine.state.current_player != 0:
            opp_reward, opp_done = self._play_opponents()
            reward += opp_reward
            terminated = opp_done

        self._refresh_legal_actions()
        obs = self._obs()
        info = dict(res.info)
        info["action_mask"] = self._action_mask()
        info["legal_actions"] = list(self._last_legal_actions)
        info["selected_action"] = selected
        info["selected_action_type"] = selected_type
        info["had_play_options_before_action"] = had_play_options
        info["phase_before_action"] = pre_phase
        info["selected_discard_project_card"] = (
            selected_type == "Discard"
            and isinstance(selected.get("card"), dict)
            and self._is_project_card_payload(selected.get("card"), pre_hand_payload)
        )
        info["selected_discard_hot_card"] = (
            selected_type == "Discard"
            and isinstance(selected.get("card"), dict)
            and self._is_hot_discard_token(self._card_token(selected.get("card")), actor_index=0)
        )
        info["episode_discard_with_play_options"] = self._episode_discard_with_play_options
        info["episode_play_when_available"] = self._episode_play_when_available
        info["episode_play_options_turns"] = self._episode_play_options_turns
        info["episode_play_when_available_ratio"] = (
            float(self._episode_play_when_available) / float(self._episode_play_options_turns)
            if self._episode_play_options_turns > 0
            else 0.0
        )

        if obs.shape[0] != self.observation_space.shape[0]:
            raise ValueError(
                f"RemoteLikeSmartLobaEnv observation shape mismatch: got {obs.shape[0]}, expected {self.observation_space.shape[0]}"
            )
        return obs, reward, terminated, False, info

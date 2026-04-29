from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from loba_ai.agents.random_agent import RandomAgent
from loba_ai.remote.adapter import build_remote_obs_vector
from loba_ai.remote_like_engine import RemoteLikeGameEngine
from loba_ai.remote_like_obs_builder import RUN_PROJECT_MAX_RANK_GAP
from loba_ai.rules import Rules

MAX_REMOTE_LIKE_ACTIONS = 256


class RemoteLikeLobaEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, rules: Rules | None = None, seed: int | None = None, opponent: str = "random") -> None:
        super().__init__()
        self.rules = rules or Rules()
        self.engine = RemoteLikeGameEngine(self.rules, seed=seed)
        self.rng = np.random.default_rng(seed)
        self.random_agent = RandomAgent(self.rng)
        self.opponent_type = opponent
        self._last_legal_actions: list[dict] = []
        self.reward_play_action_bonus = 0.65
        self.reward_extend_action_bonus = 0.45
        self.reward_cruzar_action_bonus = 0.35
        self.reward_discard_with_play_options_penalty = 1.05
        self.reward_discard_project_penalty = 0.45
        self.reward_discard_low_single_before_high_single_penalty = 0.2
        # Fixed obs shape for VecEnv/SB3 buffers: 111 + 2*num_players
        # hand(53) + top_discard(53) + counts(P) + opened(P) + stock(1) + phase(3) + pad(1)
        obs_dim = 111 + (2 * int(self.rules.num_players))
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(MAX_REMOTE_LIKE_ACTIONS)

    def _card_payload(self, card) -> dict:
        if card.is_joker:
            return {"joker": True, "deck_id": card.deck_id}
        rank_map = {1: "A", 11: "J", 12: "Q", 13: "K"}
        suit_map = {"spades": "S", "hearts": "H", "clubs": "C", "diamonds": "D"}
        return {"rank": rank_map.get(card.rank, str(card.rank)), "suit": suit_map.get(card.suit or "clubs", "C"), "deck_id": card.deck_id}

    def _build_observation_payload(self) -> dict:
        s = self.engine.state
        seat = s.current_player
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

    def _obs(self) -> np.ndarray:
        payload = self._build_observation_payload()
        return build_remote_obs_vector(payload)

    def _refresh_legal_actions(self) -> None:
        self._last_legal_actions = self.engine.legal_actions()

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        n = min(len(self._last_legal_actions), self.action_space.n)
        if n:
            mask[:n] = 1
        return mask

    def action_masks(self) -> np.ndarray:
        return self._action_mask().astype(bool)

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
        # Pair/triple project OR same-suit run project, including one-card gaps.
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

    @staticmethod
    def _has_non_discard_play_options(legal_actions: list[dict]) -> bool:
        playish = {"LayPierna", "LayEscalera", "ExtendMeld", "MoveJoker", "Cruzar"}
        return any(str(a.get("type")) in playish for a in legal_actions)

    def _play_opponents(self) -> tuple[float, bool]:
        total_reward = 0.0
        while not self.engine.state.finished and self.engine.state.current_player != 0:
            self._refresh_legal_actions()
            if not self._last_legal_actions:
                # No legal actions for rival: terminate episode conservatively to avoid state corruption.
                return total_reward - 0.5, True
            opp_idx = int(self.random_agent.act(self._action_mask()))
            opp_idx = max(0, min(opp_idx, len(self._last_legal_actions) - 1))
            action = self._last_legal_actions[opp_idx]
            res = self.engine.step(action, is_agent_player=False)
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
        self._refresh_legal_actions()
        obs = self._obs()
        if obs.shape[0] != self.observation_space.shape[0]:
            raise ValueError(
                f"RemoteLikeLobaEnv observation shape mismatch: got {obs.shape[0]}, expected {self.observation_space.shape[0]}"
            )
        return obs, {"action_mask": self._action_mask(), "legal_actions": list(self._last_legal_actions)}

    def step(self, action: int):
        bootstrap_reward = 0.0
        if self.engine.state.current_player != 0:
            # SB3 may call step while rivals are pending after previous transition.
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
        res = self.engine.step(selected, is_agent_player=True)
        reward = float(res.reward) + bootstrap_reward
        selected_type = str(selected.get("type", "Unknown"))
        had_play_options = self._has_non_discard_play_options(pre_legal_actions)

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
        if obs.shape[0] != self.observation_space.shape[0]:
            raise ValueError(
                f"RemoteLikeLobaEnv observation shape mismatch: got {obs.shape[0]}, expected {self.observation_space.shape[0]}"
            )
        return obs, reward, terminated, False, info

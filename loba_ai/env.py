from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from loba_ai.agents.heuristic_agent import HeuristicAgent
from loba_ai.agents.random_agent import RandomAgent
from loba_ai.engine import LobaGameEngine
from loba_ai.melds import discard_take_melds, find_all_melds
from loba_ai.rules import Rules

MAX_MELD_ACTIONS = 24


class LobaEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, rules: Rules | None = None, seed: int | None = None, opponent: str = "heuristic") -> None:
        super().__init__()
        self.rules = rules or Rules()
        self.engine = LobaGameEngine(self.rules, seed=seed)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(115,), dtype=np.float32)
        self.action_space = spaces.Discrete(2 + 1 + MAX_MELD_ACTIONS + self.rules.max_hand_size)
        self.rng = np.random.default_rng(seed)
        self.random_opponent = RandomAgent(self.rng)
        self.heuristic_opponent = HeuristicAgent(self.rng)
        self.opponent_type = opponent

    def _encode_cards(self, cards) -> np.ndarray:
        vec = np.zeros(53, dtype=np.float32)
        for c in cards:
            vec[c.token] += 1.0
        return vec / 4.0

    def _phase_one_hot(self, phase: str) -> np.ndarray:
        v = np.zeros(3, dtype=np.float32)
        mapping = {"draw": 0, "meld": 1, "discard": 2}
        v[mapping[phase]] = 1.0
        return v

    def _obs(self) -> np.ndarray:
        s = self.engine.state
        me = s.players[0]
        top_discard = self._encode_cards([s.discard_pile[-1]]) if s.discard_pile else np.zeros(53, dtype=np.float32)
        hand = self._encode_cards(me.hand)

        counts = np.array([len(p.hand) for p in s.players], dtype=np.float32)
        counts = counts / float(self.rules.max_hand_size)

        opened = np.array([1.0 if p.has_opened else 0.0 for p in s.players], dtype=np.float32)
        stock_size = np.array([len(s.stock_pile) / 108.0], dtype=np.float32)
        phase = self._phase_one_hot(s.phase)

        obs = np.concatenate([hand, top_discard, counts, opened, stock_size, phase, np.zeros(1, dtype=np.float32)])
        return obs.astype(np.float32)

    def _action_mask(self) -> np.ndarray:
        s = self.engine.state
        mask = np.zeros(self.action_space.n, dtype=np.int8)

        if s.phase == "draw":
            mask[0] = 1
            if s.discard_pile:
                if self.rules.must_meld_if_draw_discard:
                    top = s.discard_pile[-1]
                    hand = s.players[s.current_player].hand
                    if discard_take_melds(hand, top, self.rules, max_results=MAX_MELD_ACTIONS):
                        mask[1] = 1
                else:
                    mask[1] = 1

        elif s.phase == "meld":
            mask[2] = 1
            melds = find_all_melds(s.players[s.current_player].hand, self.rules, max_results=MAX_MELD_ACTIONS)
            for i in range(len(melds)):
                mask[3 + i] = 1

        elif s.phase == "discard":
            base = 3 + MAX_MELD_ACTIONS
            hand_len = len(s.players[s.current_player].hand)
            for i in range(min(hand_len, self.rules.max_hand_size)):
                mask[base + i] = 1

        return mask

    def action_masks(self) -> np.ndarray:
        """Compatibility hook required by sb3-contrib MaskablePPO."""
        return self._action_mask().astype(bool)

    def _play_opponent_until_agent(self) -> tuple[float, bool]:
        total_reward = 0.0
        while not self.engine.state.finished and self.engine.state.current_player != 0:
            mask = self._action_mask()
            phase = self.engine.state.phase
            if self.opponent_type == "random":
                action = self.random_opponent.act(mask)
            else:
                action = self.heuristic_opponent.act(mask, phase)
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
        self.engine.reset()
        return self._obs(), {"action_mask": self._action_mask()}

    def step(self, action: int):
        state_before = self.engine.state
        if self.opponent_type != "manual" and state_before.current_player != 0:
            raise RuntimeError("Env step called when not agent turn")

        is_agent = state_before.current_player == 0
        res = self.engine.step(action, is_agent_player=is_agent)
        reward = res.reward
        terminated = res.done

        if self.opponent_type != "manual" and not terminated and self.engine.state.current_player != 0:
            opp_reward, opp_done = self._play_opponent_until_agent()
            reward += opp_reward
            terminated = opp_done

        obs = self._obs()
        info = dict(res.info)
        info["action_mask"] = self._action_mask()
        truncated = False
        return obs, float(reward), terminated, truncated, info

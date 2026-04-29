from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sb3_contrib import MaskablePPO

from loba_ai.agents.heuristic_agent import HeuristicAgent
from loba_ai.agents.random_agent import RandomAgent
from loba_ai.cards import hand_points
from loba_ai.engine import LobaGameEngine
from loba_ai.env import MAX_MELD_ACTIONS
from loba_ai.melds import discard_take_melds, find_all_melds
from loba_ai.model_io import choose_action
from loba_ai.rules import Rules


class MatchLobaEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        rules: Rules | None = None,
        seed: int | None = None,
        opponent: str = "heuristic",
        table_mode: str = "classic",
        trained_opponent_model: MaskablePPO | None = None,
        target_points: int = 100,
        max_rounds_per_match: int = 64,
        round_score_delta_coef: float = 5.0,
        match_win_bonus: float = 150.0,
        match_loss_penalty: float = 150.0,
        meld_joker_penalty: float = 0.35,
        avoidable_joker_extra_penalty: float = 0.45,
        joker_discard_penalty: float = 0.75,
        joker_hold_bonus: float = 0.05,
    ) -> None:
        super().__init__()
        base_rules = rules or Rules()
        self.table_mode = table_mode
        if self.table_mode == "mixed4p":
            self.rules = replace(base_rules, num_players=4)
            if trained_opponent_model is None:
                raise ValueError("table_mode='mixed4p' requires a trained_opponent_model")
        else:
            self.rules = base_rules

        self.engine = LobaGameEngine(self.rules, seed=seed)
        obs_dim = 53 + 53 + (2 * self.rules.num_players) + 1 + 3 + 1 + 3
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(2 + 1 + MAX_MELD_ACTIONS + self.rules.max_hand_size)

        self.rng = np.random.default_rng(seed)
        self.random_opponent = RandomAgent(self.rng)
        self.heuristic_opponent = HeuristicAgent(self.rng)
        self.opponent_type = opponent
        self.trained_opponent_model = trained_opponent_model

        self.target_points = max(1, int(target_points))
        self.max_rounds_per_match = max(1, int(max_rounds_per_match))
        self.round_score_delta_coef = float(round_score_delta_coef)
        self.match_win_bonus = float(match_win_bonus)
        self.match_loss_penalty = float(match_loss_penalty)
        self.meld_joker_penalty = float(meld_joker_penalty)
        self.avoidable_joker_extra_penalty = float(avoidable_joker_extra_penalty)
        self.joker_discard_penalty = float(joker_discard_penalty)
        self.joker_hold_bonus = float(joker_hold_bonus)

        self.match_scores = np.zeros(self.rules.num_players, dtype=np.float32)
        self.rounds_played = 0
        self._reset_joker_telemetry()

    def _reset_joker_telemetry(self) -> None:
        self.joker_stats = {
            "melds_total": 0,
            "melds_with_joker": 0,
            "avoidable_joker_melds": 0,
            "joker_discards": 0,
            "agent_turns": 0,
            "agent_turns_holding_joker": 0,
        }

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

    def _obs_for_player(self, player_index: int) -> np.ndarray:
        s = self.engine.state
        me = s.players[player_index]
        top_discard = self._encode_cards([s.discard_pile[-1]]) if s.discard_pile else np.zeros(53, dtype=np.float32)
        hand = self._encode_cards(me.hand)

        order = list(range(player_index, len(s.players))) + list(range(0, player_index))
        counts = np.array([len(s.players[i].hand) for i in order], dtype=np.float32)
        counts = counts / float(self.rules.max_hand_size)
        opened = np.array([1.0 if s.players[i].has_opened else 0.0 for i in order], dtype=np.float32)
        stock_size = np.array([len(s.stock_pile) / 108.0], dtype=np.float32)
        phase = self._phase_one_hot(s.phase)

        rotated_scores = self.match_scores[order]
        my_score = float(rotated_scores[0]) / float(self.target_points)
        other_score = float(np.min(rotated_scores[1:])) / float(self.target_points) if len(rotated_scores) > 1 else 0.0
        score_gap = (other_score - my_score)
        match_features = np.array([my_score, other_score, score_gap], dtype=np.float32)

        obs = np.concatenate([hand, top_discard, counts, opened, stock_size, phase, np.zeros(1, dtype=np.float32), match_features])
        return obs.astype(np.float32)

    def _obs_for_player_with_dim(self, player_index: int, expected_dim: int) -> np.ndarray:
        obs = self._obs_for_player(player_index)
        if obs.shape[0] == expected_dim:
            return obs
        if expected_dim == 118 and self.rules.num_players >= 2:
            s = self.engine.state
            me = s.players[player_index]
            top_discard = self._encode_cards([s.discard_pile[-1]]) if s.discard_pile else np.zeros(53, dtype=np.float32)
            hand = self._encode_cards(me.hand)

            order = list(range(player_index, len(s.players))) + list(range(0, player_index))
            counts = np.array([len(s.players[i].hand) for i in order[:2]], dtype=np.float32)
            counts = counts / float(self.rules.max_hand_size)
            opened = np.array([1.0 if s.players[i].has_opened else 0.0 for i in order[:2]], dtype=np.float32)
            stock_size = np.array([len(s.stock_pile) / 108.0], dtype=np.float32)
            phase = self._phase_one_hot(s.phase)

            rotated_scores = self.match_scores[order]
            my_score = float(rotated_scores[0]) / float(self.target_points)
            other_score = float(rotated_scores[1]) / float(self.target_points)
            score_gap = other_score - my_score
            match_features = np.array([my_score, other_score, score_gap], dtype=np.float32)
            obs_2p = np.concatenate(
                [hand, top_discard, counts, opened, stock_size, phase, np.zeros(1, dtype=np.float32), match_features]
            )
            return obs_2p.astype(np.float32)
        raise ValueError(f"Opponent model expects unsupported obs dim {expected_dim}, env produces {obs.shape[0]}")

    def _obs(self) -> np.ndarray:
        return self._obs_for_player(0)

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
        return self._action_mask().astype(bool)

    def _choose_opponent_action(self, phase: str, mask: np.ndarray, player_index: int) -> int:
        if self.table_mode == "mixed4p":
            if player_index == 1:
                if self.trained_opponent_model is None:
                    raise RuntimeError("Missing trained opponent model for mixed4p")
                model_obs_dim = int(self.trained_opponent_model.observation_space.shape[0])
                model_obs = self._obs_for_player_with_dim(player_index, model_obs_dim)
                return choose_action(self.trained_opponent_model, model_obs, mask.astype(bool))
            if player_index == 2:
                return self.heuristic_opponent.act(mask, phase)
            if player_index == 3:
                return self.random_opponent.act(mask)
            return self.heuristic_opponent.act(mask, phase)

        if self.opponent_type == "random":
            return self.random_opponent.act(mask)
        return self.heuristic_opponent.act(mask, phase)

    def _finish_round(self) -> tuple[float, bool, dict]:
        hand_scores = np.array([hand_points(p.hand) for p in self.engine.state.players], dtype=np.float32)
        self.match_scores += hand_scores
        self.rounds_played += 1

        my_score = float(self.match_scores[0])
        other_best = float(np.min(self.match_scores[1:])) if len(self.match_scores) > 1 else my_score
        round_reward = self.round_score_delta_coef * ((other_best - my_score) / float(self.target_points))

        reached_target = bool(np.any(self.match_scores >= float(self.target_points)))
        reached_round_limit = self.rounds_played >= self.max_rounds_per_match
        done = reached_target or reached_round_limit

        info = {
            "round_over": True,
            "round_hand_points": hand_scores.tolist(),
            "match_scores": self.match_scores.tolist(),
            "rounds_played": self.rounds_played,
        }

        if done:
            winner = int(np.argmin(self.match_scores))
            info["match_winner"] = winner
            info["match_finished"] = True
            melds_total = int(self.joker_stats["melds_total"])
            melds_with_joker = int(self.joker_stats["melds_with_joker"])
            info["joker_meld_rate"] = (
                float(melds_with_joker) / float(melds_total) if melds_total > 0 else 0.0
            )
            info["joker_discards"] = int(self.joker_stats["joker_discards"])
            info["avoidable_joker_melds"] = int(self.joker_stats["avoidable_joker_melds"])
            turns_total = int(self.joker_stats["agent_turns"])
            turns_holding = int(self.joker_stats["agent_turns_holding_joker"])
            info["joker_hold_rate"] = (
                float(turns_holding) / float(turns_total) if turns_total > 0 else 0.0
            )
            if reached_round_limit and not reached_target:
                info["match_forced_termination"] = True
            if winner == 0:
                round_reward += self.match_win_bonus
            else:
                round_reward -= self.match_loss_penalty
            return round_reward, True, info

        self.engine.reset()
        return round_reward, False, info

    def _play_opponents_until_agent_or_round_end(self) -> tuple[float, bool, dict]:
        total_reward = 0.0
        info_acc: dict = {}
        while not self.engine.state.finished and self.engine.state.current_player != 0:
            mask = self._action_mask()
            phase = self.engine.state.phase
            action = self._choose_opponent_action(phase, mask, player_index=self.engine.state.current_player)
            res = self.engine.step(action, is_agent_player=False)
            if res.done:
                round_reward, done, round_info = self._finish_round()
                total_reward += round_reward
                info_acc.update(round_info)
                return total_reward, done, info_acc
        return total_reward, False, info_acc

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.random_opponent = RandomAgent(self.rng)
            self.heuristic_opponent = HeuristicAgent(self.rng)
        self.engine.reset()
        self.match_scores = np.zeros(self.rules.num_players, dtype=np.float32)
        self.rounds_played = 0
        self._reset_joker_telemetry()
        return self._obs(), {"action_mask": self._action_mask(), "match_scores": self.match_scores.tolist()}

    def step(self, action: int):
        if self.engine.state.current_player != 0:
            raise RuntimeError("MatchEnv step called when not agent turn")

        res = self.engine.step(action, is_agent_player=True)
        reward = float(res.reward)
        terminated = False
        info = dict(res.info)
        agent_hand = self.engine.state.players[0].hand
        self.joker_stats["agent_turns"] += 1
        if any(c.is_joker for c in agent_hand):
            self.joker_stats["agent_turns_holding_joker"] += 1

        if info.get("meld_size") is not None:
            self.joker_stats["melds_total"] += 1
            if info.get("meld_used_joker"):
                self.joker_stats["melds_with_joker"] += 1
                reward -= self.meld_joker_penalty * float(info.get("meld_jokers_used", 1))
                if int(info.get("meld_natural_alternatives", 0)) > 0:
                    self.joker_stats["avoidable_joker_melds"] += 1
                    reward -= self.avoidable_joker_extra_penalty

        if info.get("discard_is_joker"):
            self.joker_stats["joker_discards"] += 1
            reward -= self.joker_discard_penalty
        elif info.get("discard_is_joker") is False and any(c.is_joker for c in agent_hand):
            reward += self.joker_hold_bonus

        if res.done:
            round_reward, terminated, round_info = self._finish_round()
            reward += round_reward
            info.update(round_info)
        elif self.engine.state.current_player != 0:
            opp_reward, opp_done, opp_info = self._play_opponents_until_agent_or_round_end()
            reward += opp_reward
            terminated = opp_done
            info.update(opp_info)

        obs = self._obs()
        info["action_mask"] = self._action_mask()
        info["match_scores"] = self.match_scores.tolist()
        truncated = False
        return obs, float(reward), terminated, truncated, info

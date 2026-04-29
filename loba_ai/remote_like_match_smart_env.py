from __future__ import annotations

from typing import Any

from collections import deque
from pathlib import Path

import numpy as np
from gymnasium import spaces

from loba_ai.agents.remote_like_heuristic_agent import RemoteLikeHeuristicAgent, StrongRemoteLikeHeuristicAgent
from loba_ai.cards import hand_points
from loba_ai.remote_like_obs_builder import (
    OWN_ACTION_HISTORY_LEN,
    SmartObsMemory,
    build_smart_match_obs,
    describe_action_tactics,
    smart_match_obs_dim,
)
from loba_ai.remote_like_smart_env import RemoteLikeSmartLobaEnv
from loba_ai.rules import Rules


class RemoteLikeMatchSmartLobaEnv(RemoteLikeSmartLobaEnv):
    """Match-aware variant: an episode spans multiple rounds until target_points is reached."""

    def __init__(
        self,
        rules: Rules | None = None,
        seed: int | None = None,
        opponent: str = "random",
        discard_history_window: int = 2,
        target_points: int = 100,
        max_rounds_per_match: int = 64,
        round_score_delta_coef: float = 5.0,
        match_win_bonus: float = 150.0,
        match_loss_penalty: float = 150.0,
        round_win_terminal: float = 10.0,
        round_loss_terminal_coef: float = 0.1,
        max_reenganches: int = 2,
        reward_unsafe_meld_penalty: float = 0.0,
        reward_discard_project_with_high_dead_penalty: float = 0.0,
        reward_discard_high_dead_bonus: float = 0.0,
        reward_discard_break_project_penalty: float = 0.0,
        enable_action_tactical_features: bool = True,
        trained_opponent_model: Any = None,
        opponent_model_seats: str = "all",
        self_play_pool_dir: str | None = None,
        self_play_sample_prob: float = 1.0,
    ) -> None:
        super().__init__(
            rules=rules,
            seed=seed,
            opponent=opponent,
            discard_history_window=discard_history_window,
        )

        self.target_points = max(1, int(target_points))
        self.max_rounds_per_match = max(1, int(max_rounds_per_match))
        self.round_score_delta_coef = float(round_score_delta_coef)
        self.match_win_bonus = float(match_win_bonus)
        self.match_loss_penalty = float(match_loss_penalty)
        self.round_win_terminal = float(round_win_terminal)
        self.round_loss_terminal_coef = float(round_loss_terminal_coef)
        self.max_reenganches = max(0, int(max_reenganches))
        self.reward_unsafe_meld_penalty = float(reward_unsafe_meld_penalty)
        self.reward_discard_project_with_high_dead_penalty = float(
            reward_discard_project_with_high_dead_penalty
        )
        self.reward_discard_high_dead_bonus = float(reward_discard_high_dead_bonus)
        self.reward_discard_break_project_penalty = float(
            reward_discard_break_project_penalty
        )
        self.enable_action_tactical_features = bool(enable_action_tactical_features)
        self.trained_opponent_model = trained_opponent_model
        self.opponent_model_seats = str(opponent_model_seats)
        # Self-play snapshot pool. If set, _maybe_sample_self_play_opponent() picks a
        # random snapshot from this directory at each reset and assigns it as the
        # opponent model (overriding the constructor-provided one for that match).
        self.self_play_pool_dir: str | None = (
            str(self_play_pool_dir) if self_play_pool_dir else None
        )
        self.self_play_sample_prob: float = float(self_play_sample_prob)
        # Remember the constructor-provided opponent so we can fall back to it when
        # the pool is empty or the dice say "skip self-play this match".
        self._base_opponent_model: Any = trained_opponent_model
        # Cache loaded snapshots so we don't reload from disk on every reset.
        # Keyed by (path, mtime) so we pick up freshly-saved snapshots.
        self._self_play_model_cache: dict[tuple[str, float], Any] = {}

        self.match_scores = np.zeros(self.rules.num_players, dtype=np.float32)
        self.reenganches_used = np.zeros(self.rules.num_players, dtype=np.int32)
        self.eliminated = np.zeros(self.rules.num_players, dtype=bool)
        self.rounds_played = 0
        self._match_round_records: list[dict[str, Any]] = []
        self._match_committed_winner: int | None = None
        self._actions_in_match = 0
        # Round number (1-indexed) at which the agent (seat 0) was eliminated. None if still active
        # at end of episode.
        self._agent_elimination_round: int | None = None
        self._last_action_tactics: dict[str, Any] | None = None
        self.max_actions_per_match = 5000  # safety cap, way above realistic worst-case

        self.heuristic_agent = (
            RemoteLikeHeuristicAgent(rng=self.rng)
            if opponent in {"heuristic", "mixed_heuristic"}
            else None
        )
        self.strong_heuristic_agent = (
            StrongRemoteLikeHeuristicAgent(rng=self.rng)
            if opponent in {"strong_heuristic", "mixed_heuristic"}
            else None
        )

        # Full smart obs: parent base + match + table + stock_remaining + own_actions.
        # See loba_ai/remote_like_obs_builder.smart_match_obs_dim for the formula.
        full_obs_dim = smart_match_obs_dim(
            num_players=int(self.rules.num_players),
            discard_history_window=int(self.discard_history_window),
            include_action_tactics=bool(self.enable_action_tactical_features),
        )
        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(full_obs_dim,),
            dtype=np.float32,
        )

        # Track our own action types for the own-action-history obs feature.
        self._own_action_history: deque[str] = deque(maxlen=OWN_ACTION_HISTORY_LEN)

    # ------------------------------------------------------------------ #
    # Match-aware observation
    # ------------------------------------------------------------------ #
    def _match_features_for_seat(self, seat: int) -> np.ndarray:
        n = self.rules.num_players
        order = list(range(seat, n)) + list(range(0, seat))
        rotated_scores = self.match_scores[order]
        rotated_reeng = self.reenganches_used[order]
        rotated_elim = self.eliminated[order]
        scores_norm = (rotated_scores / float(self.target_points)).astype(np.float32)
        reeng_norm = (
            rotated_reeng.astype(np.float32) / max(1.0, float(self.max_reenganches))
        ).astype(np.float32)
        elim_norm = rotated_elim.astype(np.float32)
        my_score = float(scores_norm[0])
        active_others = [
            float(scores_norm[i]) for i in range(1, n) if not bool(rotated_elim[i])
        ]
        if active_others:
            best_other = min(active_others)
        else:
            best_other = my_score
        gap = best_other - my_score
        # Layout: [my_score, opp1_score, opp2_score, ..., my_reeng, opp1_reeng, ..., my_elim, opp1_elim, ..., gap_to_best]
        return np.concatenate(
            [scores_norm, reeng_norm, elim_norm, np.array([gap], dtype=np.float32)]
        ).astype(np.float32)

    def _table_features_for_seat(self, seat: int) -> np.ndarray:
        """Encode the public state of melds on the table from the viewer's perspective.

        Layout (3p): 53 card-presence + 3 cards-laid-by-owner + 3 melds-by-owner + 4 aggregates.
        Total = 63 dims for 3 players.
        """
        n = self.rules.num_players
        melds = self.engine.table_melds

        card_presence = np.zeros(TABLE_FEATURE_BASE, dtype=np.float32)
        cards_laid = np.zeros(n, dtype=np.float32)
        melds_per_owner = np.zeros(n, dtype=np.float32)
        escalera_count = 0
        pierna_count = 0
        escalera_left_joker = 0
        escalera_right_joker = 0

        for meld in melds:
            owner = int(meld.get("owner", 0))
            rotated_owner = (owner - seat) % n  # rotate so viewer maps to slot 0
            cards = meld.get("cards", [])
            if not isinstance(cards, list):
                continue
            kind = str(meld.get("kind", ""))

            for c in cards:
                token = int(getattr(c, "token", 52))
                if 0 <= token < TABLE_FEATURE_BASE:
                    card_presence[token] += 1.0

            cards_laid[rotated_owner] += float(len(cards))
            melds_per_owner[rotated_owner] += 1.0

            if kind == "escalera":
                escalera_count += 1
                if cards and bool(getattr(cards[0], "is_joker", False)):
                    escalera_left_joker += 1
                if cards and bool(getattr(cards[-1], "is_joker", False)):
                    escalera_right_joker += 1
            elif kind == "pierna":
                pierna_count += 1

        # Normalize: max 4 copies of any natural (2 decks * 2 = at most 4 in a meld).
        card_presence = np.clip(card_presence / 4.0, 0.0, 1.0)
        cards_laid_norm = np.clip(cards_laid / 14.0, 0.0, 1.0)  # ~14 cards = aggressive opener
        melds_norm = np.clip(melds_per_owner / 5.0, 0.0, 1.0)

        aggregates = np.array(
            [
                min(1.0, escalera_count / 8.0),
                min(1.0, pierna_count / 8.0),
                min(1.0, escalera_left_joker / 4.0),
                min(1.0, escalera_right_joker / 4.0),
            ],
            dtype=np.float32,
        )

        return np.concatenate(
            [card_presence, cards_laid_norm, melds_norm, aggregates]
        ).astype(np.float32)

    def _obs(self) -> np.ndarray:
        # Build via the shared obs module so training and deployment use identical layouts.
        payload = self._build_observation_payload(seat=0)
        memory = self._snapshot_memory()
        return build_smart_match_obs(
            payload,
            memory,
            melds=self.engine.table_melds,
            legal_actions=self._last_legal_actions,
            include_action_tactics=bool(self.enable_action_tactical_features),
        )

    def _snapshot_memory(self) -> SmartObsMemory:
        """Wrap the env's mutable state into a SmartObsMemory the shared builder expects."""
        return SmartObsMemory(
            num_players=int(self.rules.num_players),
            discard_history_window=int(self.discard_history_window),
            target_points=int(self.target_points),
            max_reenganches=int(self.max_reenganches),
            seen_token_counts=self._seen_token_counts,
            recent_discards_by_player=self._recent_discards_by_player,
            own_action_history=self._own_action_history,
            match_scores=self.match_scores,
            reenganches_used=self.reenganches_used,
            eliminated=self.eliminated,
        )

    def _opponent_obs(self, seat: int) -> np.ndarray:
        # Opponent model (e.g. v12) was trained without match/table features → keep 385-dim parent obs.
        return self._build_obs_for_seat(seat=int(seat))

    # ------------------------------------------------------------------ #
    # Opponent loop with optional self-play model
    # ------------------------------------------------------------------ #
    def _opponent_select_action(self, seat: int) -> int:
        mask = self._action_mask()
        if self._should_use_trained_opponent(seat):
            try:
                obs = self._opponent_obs(seat)
                action, _ = self.trained_opponent_model.predict(
                    obs, deterministic=False, action_masks=mask.astype(bool)
                )
                idx = int(action)
                if 0 <= idx < len(self._last_legal_actions):
                    return idx
            except Exception:
                pass
        heuristic_agent = self._heuristic_agent_for_seat(seat)
        if heuristic_agent is not None:
            hand_payload = [self._card_payload(c) for c in self.engine.state.players[seat].hand]
            phase = str(self.engine.state.phase)
            try:
                idx = int(heuristic_agent.act(self._last_legal_actions, hand_payload, phase))
                if 0 <= idx < len(self._last_legal_actions):
                    return idx
            except Exception:
                pass
        return int(self.random_agent.act(mask))

    def _should_use_trained_opponent(self, seat: int) -> bool:
        if self.trained_opponent_model is None:
            return False
        mode = self.opponent_model_seats
        if mode == "none":
            return False
        if mode == "last":
            return int(seat) == int(self.rules.num_players) - 1
        return True

    def _heuristic_agent_for_seat(self, seat: int):
        if self.opponent_type == "mixed_heuristic":
            # Seat 0 is the RL agent. In a 3p game this means player 2 uses the
            # existing heuristic and player 3 uses the stronger heuristic.
            return self.heuristic_agent if int(seat) == 1 else self.strong_heuristic_agent
        if self.opponent_type == "strong_heuristic":
            return self.strong_heuristic_agent
        return self.heuristic_agent

    def _play_opponents(self) -> tuple[float, bool]:
        """Play opponents until it's the agent's turn OR the round ends.

        Round end here does NOT terminate the episode; it triggers a match update.
        Returns (round_reward_so_far, match_terminated).
        """
        round_reward = 0.0
        # Defensive: if the round just started and current_player is eliminated, skip.
        self._skip_eliminated_seats()
        while not self.engine.state.finished and self.engine.state.current_player != 0:
            self._actions_in_match += 1
            if self._actions_in_match > self.max_actions_per_match:
                round_reward += self._on_round_end_attenuated(winner_seat=None)
                match_done, match_reward = self._maybe_finish_match(forced=True)
                round_reward += match_reward
                return round_reward, True
            self._refresh_legal_actions()
            if not self._last_legal_actions:
                # Degenerate: no legal action for opponent → end round defensively.
                round_reward += self._on_round_end_attenuated(winner_seat=None)
                match_done, match_reward = self._maybe_finish_match(forced=True)
                round_reward += match_reward
                return round_reward, match_done

            opp_idx = self._opponent_select_action(self.engine.state.current_player)
            opp_idx = max(0, min(opp_idx, len(self._last_legal_actions) - 1))
            action = self._last_legal_actions[opp_idx]
            actor_index = int(self.engine.state.current_player)
            self._track_move_joker_guard(actor_index, action)
            res = self.engine.step(action, is_agent_player=False)
            self._record_discard_event(actor_index, action, not bool(res.info.get("invalid_action")))
            self._track_public_seen_cards()

            if res.done:
                winner = self.engine.state.winner if self.engine.state.winner is not None else None
                round_reward += self._on_round_end_attenuated(winner_seat=winner)
                match_done, match_reward = self._maybe_finish_match()
                round_reward += match_reward
                if match_done:
                    return round_reward, True
                # Round ended but match continues → reset engine and keep playing.
                self._reset_engine_for_next_round()
                if self.engine.state.current_player == 0:
                    return round_reward, False
                # else: keep looping with new round for opponents
            else:
                # Engine may have rotated current_player to a now-eliminated seat after a Discard.
                self._skip_eliminated_seats()
        return round_reward, False

    # ------------------------------------------------------------------ #
    # Round/match transitions
    # ------------------------------------------------------------------ #
    def _on_round_end_attenuated(self, winner_seat: int | None) -> float:
        """Attenuated terminal reward for the round + score-delta shaping. Updates match_scores."""
        raw_hand_scores = np.array(
            [hand_points(p.hand) for p in self.engine.state.players],
            dtype=np.float32,
        )
        # Eliminated players' scores are frozen — they don't accumulate hand points anymore.
        delta = np.where(self.eliminated, 0.0, raw_hand_scores)
        self.match_scores += delta
        self.rounds_played += 1

        reenganche_events = self._apply_reenganches(round_winner=winner_seat)

        self._match_round_records.append(
            {
                "round_index": int(self.rounds_played),
                "hand_scores": raw_hand_scores.tolist(),
                "match_scores": self.match_scores.tolist(),
                "winner_seat": int(winner_seat) if winner_seat is not None else None,
                "eliminated": self.eliminated.tolist(),
                "reenganches_used": self.reenganches_used.tolist(),
                "reenganche_events": reenganche_events,
            }
        )

        reward = 0.0
        if winner_seat == 0:
            reward += self.round_win_terminal
        else:
            agent_hand_pts = float(raw_hand_scores[0])
            reward -= self.round_loss_terminal_coef * agent_hand_pts

        # Score-delta shaping: only consider non-eliminated opponents.
        my_score = float(self.match_scores[0])
        active_other_scores = [
            float(self.match_scores[i])
            for i in range(1, self.rules.num_players)
            if not bool(self.eliminated[i])
        ]
        best_other = min(active_other_scores) if active_other_scores else my_score
        score_delta = (best_other - my_score) / float(self.target_points)
        reward += self.round_score_delta_coef * score_delta
        return float(reward)

    def _apply_reenganches(self, round_winner: int | None) -> list[dict[str, Any]]:
        """After scores update, eliminate or reenganche players that crossed the target.

        A player at seat i with score >= target attempts a reenganche. It succeeds iff:
          (a) reenganches_used[i] < max_reenganches, AND
          (b) at least one OTHER non-eliminated player who is NOT the round winner has score < target.

        On success, score[i] is reset to the highest score among non-eliminated, non-winner players
        that are below target (i.e. they "tie up" with the closest live competitor under 100).
        On failure, eliminated[i] = True and score is frozen.
        """
        events: list[dict[str, Any]] = []
        target = float(self.target_points)
        n = self.rules.num_players

        # Collect candidates first; apply sequentially in seat order.
        for seat in range(n):
            if self.eliminated[seat]:
                continue
            if self.match_scores[seat] < target:
                continue

            # Compute the reference set: non-eliminated players, not the round winner, score < target.
            ref_scores = [
                float(self.match_scores[j])
                for j in range(n)
                if j != seat
                and not bool(self.eliminated[j])
                and j != round_winner
                and self.match_scores[j] < target
            ]
            can_reenganche = (
                self.reenganches_used[seat] < self.max_reenganches and len(ref_scores) > 0
            )

            if can_reenganche:
                new_score = float(max(ref_scores))
                events.append(
                    {
                        "seat": seat,
                        "kind": "reenganche",
                        "from": float(self.match_scores[seat]),
                        "to": new_score,
                        "reenganches_used": int(self.reenganches_used[seat]) + 1,
                    }
                )
                self.match_scores[seat] = new_score
                self.reenganches_used[seat] += 1
            else:
                events.append(
                    {
                        "seat": seat,
                        "kind": "eliminated",
                        "score": float(self.match_scores[seat]),
                        "reason": (
                            "no_reenganches_left"
                            if self.reenganches_used[seat] >= self.max_reenganches
                            else "no_eligible_reference"
                        ),
                    }
                )
                self.eliminated[seat] = True
                if seat == 0 and self._agent_elimination_round is None:
                    self._agent_elimination_round = int(self.rounds_played)

        return events

    def _unsafe_meld_penalty(self, selected: dict[str, Any], selected_type: str, round_done: bool) -> float:
        """Penalize non-finishing melds that leave us bustable while already near 100.

        Loba table intuition: the danger zone is dynamic. Laying a meld is unsafe
        when the remaining hand would put us over target if another player closes.
        """
        if self.reward_unsafe_meld_penalty <= 0.0 or round_done:
            return 0.0

        play_type = selected_type
        if selected_type == "DrawDiscard":
            nested = selected.get("play")
            if not isinstance(nested, dict):
                return 0.0
            play_type = str(nested.get("type", ""))

        if play_type not in {"LayPierna", "LayEscalera"}:
            return 0.0

        current_score = float(self.match_scores[0])
        remaining_hand_points = float(hand_points(self.engine.state.players[0].hand))
        unsafe_excess = (current_score + remaining_hand_points) - float(self.target_points)
        if unsafe_excess <= 0.0:
            return 0.0

        return -float(self.reward_unsafe_meld_penalty) * (unsafe_excess / max(1.0, float(self.target_points)))

    def _maybe_finish_match(self, forced: bool = False) -> tuple[bool, float]:
        """Check if the match has ended. From the agent's perspective the match is over when:
          - the agent (seat 0) is eliminated → immediate loss; OR
          - at most 1 active player remains; OR
          - max_rounds_per_match reached / explicitly forced.

        Returns (terminated, extra_reward).
        """
        active_mask = ~self.eliminated
        n_active = int(np.sum(active_mask))

        # Agent eliminated → match is over for our training purposes (no point
        # collecting more transitions on a lost cause).
        if bool(self.eliminated[0]):
            if n_active >= 1:
                masked = np.where(self.eliminated, np.inf, self.match_scores)
                winner = int(np.argmin(masked))
            else:
                winner = int(np.argmin(self.match_scores))
            self._match_committed_winner = winner
            return True, -float(self.match_loss_penalty)

        reached_round_cap = self.rounds_played >= self.max_rounds_per_match
        match_over = n_active <= 1 or reached_round_cap or forced
        if not match_over:
            return False, 0.0

        if n_active == 1:
            winner = int(np.argmax(active_mask))  # index of the only True
        elif n_active >= 2:
            # Forced or round-cap with multiple active → lowest among active wins.
            masked = np.where(self.eliminated, np.inf, self.match_scores)
            winner = int(np.argmin(masked))
        else:
            # 0 active (all eliminated same round) — lowest absolute score wins.
            winner = int(np.argmin(self.match_scores))

        self._match_committed_winner = winner
        if winner == 0:
            return True, float(self.match_win_bonus)
        return True, -float(self.match_loss_penalty)

    def _reset_engine_for_next_round(self) -> None:
        """Re-deal a new round; rotate the starting seat. Eliminated players don't
        get a hand and are skipped in the turn rotation."""
        self.engine.reset()
        n = self.rules.num_players
        # Eliminated players: return their dealt cards to the stock and shuffle so
        # the deck count stays consistent and they hold no cards.
        any_eliminated = False
        for seat in range(n):
            if bool(self.eliminated[seat]):
                any_eliminated = True
                self.engine.state.stock_pile.extend(self.engine.state.players[seat].hand)
                self.engine.state.players[seat].hand = []
        if any_eliminated:
            self.engine.rng.shuffle(self.engine.state.stock_pile)
        # Pick starting seat: nominal rotation, then advance past any eliminated seats.
        desired = int(self.rounds_played % n)
        safety = n
        while safety > 0 and bool(self.eliminated[desired]):
            desired = (desired + 1) % n
            safety -= 1
        self.engine.state.current_player = desired
        self.engine.state.phase = "draw"
        # Reset per-round private memory.
        self._seen_token_counts.fill(0.0)
        self._seen_unique_ids = set()
        self._recent_discards_by_player = [[] for _ in range(self.rules.num_players)]
        self._consecutive_move_jokers = 0
        self._last_actor_for_move_joker_guard = None
        self._own_action_history.clear()
        self._track_public_seen_cards()
        self._refresh_legal_actions()

    # ------------------------------------------------------------------ #
    # Self-play snapshot pool
    # ------------------------------------------------------------------ #
    def _maybe_sample_self_play_opponent(self) -> None:
        """If a snapshot pool dir is configured, pick a random .zip from it and
        load it as `trained_opponent_model` for this match. With probability
        (1 - self_play_sample_prob), restore the original opponent instead.
        No-op when the pool dir is unset or empty.
        """
        if not self.self_play_pool_dir:
            return
        try:
            pool_path = Path(self.self_play_pool_dir)
            if not pool_path.exists():
                self.trained_opponent_model = self._base_opponent_model
                return
            snapshots = sorted(pool_path.glob("*.zip"), key=lambda p: p.stat().st_mtime)
        except Exception:
            self.trained_opponent_model = self._base_opponent_model
            return
        if not snapshots:
            self.trained_opponent_model = self._base_opponent_model
            return
        # With prob (1 - sample_prob), skip self-play this match and use the
        # base opponent (random/heuristic/etc).
        if self.rng.random() > self.self_play_sample_prob:
            self.trained_opponent_model = self._base_opponent_model
            return
        chosen = snapshots[int(self.rng.integers(0, len(snapshots)))]
        try:
            mtime = chosen.stat().st_mtime
            cache_key = (str(chosen), mtime)
            cached = self._self_play_model_cache.get(cache_key)
            if cached is None:
                from loba_ai.model_io import load_model

                cached = load_model(str(chosen))
                # Keep cache small (paths rotate via FIFO eviction in trainer).
                if len(self._self_play_model_cache) > 6:
                    # Drop oldest cached entry.
                    oldest = next(iter(self._self_play_model_cache))
                    self._self_play_model_cache.pop(oldest, None)
                self._self_play_model_cache[cache_key] = cached
            self.trained_opponent_model = cached
        except Exception:
            self.trained_opponent_model = self._base_opponent_model

    def _skip_eliminated_seats(self) -> None:
        """If the engine landed on an eliminated seat (e.g. after an end-of-turn
        rotation), advance to the next active seat and reset its phase to 'draw'.
        No-op when the current seat is active or the round is already finished."""
        if self.engine.state.finished:
            return
        n = self.rules.num_players
        safety = n + 1
        while safety > 0 and bool(self.eliminated[int(self.engine.state.current_player)]):
            self.engine.state.current_player = (self.engine.state.current_player + 1) % n
            self.engine.state.phase = "draw"
            safety -= 1

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Refresh the self-play opponent BEFORE any opponents act this match.
        self._maybe_sample_self_play_opponent()
        obs, info = super().reset(seed=seed, options=options)
        self.match_scores = np.zeros(self.rules.num_players, dtype=np.float32)
        self.reenganches_used = np.zeros(self.rules.num_players, dtype=np.int32)
        self.eliminated = np.zeros(self.rules.num_players, dtype=bool)
        self.rounds_played = 0
        self._match_round_records = []
        self._match_committed_winner = None
        self._actions_in_match = 0
        self._agent_elimination_round = None
        self._last_action_tactics = None
        self._own_action_history.clear()
        # Recompute obs to include the (zeroed) match features.
        new_obs = self._obs()
        info["match_scores"] = self.match_scores.tolist()
        info["reenganches_used"] = self.reenganches_used.tolist()
        info["eliminated"] = self.eliminated.tolist()
        info["rounds_played"] = self.rounds_played
        return new_obs, info

    def step(self, action: int):
        bootstrap_reward = 0.0

        # If it's not the agent's turn yet, autoplay opponents first (may close rounds).
        if self.engine.state.current_player != 0:
            opp_reward, match_done = self._play_opponents()
            bootstrap_reward += float(opp_reward)
            if match_done:
                self._refresh_legal_actions()
                obs = self._obs()
                info = self._build_info(autoplay=True)
                return obs, bootstrap_reward, True, False, info

        self._refresh_legal_actions()
        pre_legal_actions = list(self._last_legal_actions)
        pre_phase = str(self.engine.state.phase)
        pre_hand_payload = [self._card_payload(c) for c in self.engine.state.players[0].hand]

        if not self._last_legal_actions:
            # Agent has no legal action — should be unreachable, but keep a clean termination
            # so callbacks (win_rate etc.) get a properly-attributed match_winner via the
            # forced-finish path.
            forced_round_reward = self._on_round_end_attenuated(winner_seat=None)
            _, match_extra = self._maybe_finish_match(forced=True)
            obs = self._obs()
            info = self._build_info(autoplay=False)
            info["invalid_action"] = True
            return obs, -1.0 + bootstrap_reward + forced_round_reward + match_extra, True, False, info

        action_ix = int(action)
        if action_ix < 0 or action_ix >= len(self._last_legal_actions):
            action_ix = 0
        selected = self._last_legal_actions[action_ix]
        actor_index = int(self.engine.state.current_player)
        selected_tactics = describe_action_tactics(
            selected,
            hand_payload=pre_hand_payload,
            current_score=float(self.match_scores[0]),
            target_points=int(self.target_points),
            table_melds=self.engine.table_melds,
        )
        self._last_action_tactics = selected_tactics
        self._actions_in_match += 1
        self._track_move_joker_guard(actor_index, selected)
        # Record this action type for the own-action-history obs feature.
        self._own_action_history.append(str(selected.get("type", "")))
        res = self.engine.step(selected, is_agent_player=True)
        valid = not bool(res.info.get("invalid_action"))
        self._record_discard_event(actor_index, selected, valid)
        self._track_public_seen_cards()

        # On round close, the engine's res.reward bundles a step penalty (-0.01) with the
        # round-terminal bonus (+100 / +50 / -hand_points[0]). We replace the terminal
        # piece with our attenuated version, keeping only the small step penalty.
        if res.done:
            per_step_reward = -0.01
        else:
            per_step_reward = float(res.reward)
        reward = per_step_reward + bootstrap_reward

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
                    if bool(selected_tactics.get("is_project_card")):
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
                    if bool(selected_tactics.get("has_high_dead_alternative")):
                        reward -= self.reward_discard_project_with_high_dead_penalty
                    if bool(selected_tactics.get("breaks_project")):
                        reward -= self.reward_discard_break_project_penalty
                    if (
                        self.reward_discard_high_dead_bonus
                        and not bool(selected_tactics.get("is_project_card"))
                        and int(selected_tactics.get("card_points", 0))
                        >= int(selected_tactics.get("best_dead_card_points", 0))
                        and bool(selected_tactics.get("any_project_available"))
                    ):
                        reward += self.reward_discard_high_dead_bonus
        elif pre_phase == "draw" and selected_type == "DrawDiscard":
            # DrawDiscard always carries a nested play (LayPierna/LayEscalera/ExtendMeld/MoveJoker).
            # That play is effectively a meld action — apply the same shaping bonus so the model
            # isn't sub-incentivized to take productive cards from the discard pile.
            nested = selected.get("play")
            if isinstance(nested, dict):
                nested_type = str(nested.get("type", ""))
                if nested_type in {"LayPierna", "LayEscalera"}:
                    reward += self.reward_play_action_bonus
                elif nested_type in {"ExtendMeld", "MoveJoker"}:
                    reward += self.reward_extend_action_bonus

        reward += self._unsafe_meld_penalty(selected, selected_type, round_done=bool(res.done))

        terminated = False
        if res.done:
            # Round ended on the agent's action → apply our attenuated terminal + match update.
            winner = self.engine.state.winner if self.engine.state.winner is not None else None
            reward += self._on_round_end_attenuated(winner_seat=winner)
            match_done, match_extra = self._maybe_finish_match()
            reward += match_extra
            if match_done:
                terminated = True
            else:
                self._reset_engine_for_next_round()
        else:
            # Engine may have rotated to an eliminated opponent after a Discard.
            self._skip_eliminated_seats()

        if not terminated and self.engine.state.current_player != 0:
            opp_reward, match_done = self._play_opponents()
            reward += opp_reward
            terminated = match_done

        self._refresh_legal_actions()
        obs = self._obs()
        info = self._build_info(autoplay=False)
        info["selected_action"] = selected
        info["selected_action_type"] = selected_type
        info["had_play_options_before_action"] = had_play_options
        info["phase_before_action"] = pre_phase

        if obs.shape[0] != self.observation_space.shape[0]:
            raise ValueError(
                f"RemoteLikeMatchSmartLobaEnv obs shape mismatch: got {obs.shape[0]}, expected {self.observation_space.shape[0]}"
            )
        return obs, float(reward), terminated, False, info

    def _build_info(self, autoplay: bool) -> dict[str, Any]:
        active_mask = ~self.eliminated
        n_active = int(np.sum(active_mask))
        info: dict[str, Any] = {
            "action_mask": self._action_mask(),
            "legal_actions": list(self._last_legal_actions),
            "match_scores": self.match_scores.tolist(),
            "reenganches_used": self.reenganches_used.tolist(),
            "eliminated": self.eliminated.tolist(),
            "n_active_players": n_active,
            "rounds_played": self.rounds_played,
            "round_records": list(self._match_round_records[-3:]),
            "agent_eliminated": bool(self.eliminated[0]),
            "agent_elimination_round": self._agent_elimination_round,
            "selected_action_tactics": self._last_action_tactics,
        }
        if autoplay:
            info["autoplay_before_action"] = True
        if self.rounds_played > 0:
            # Match leader = lowest score among active (or absolute if none).
            if n_active >= 1:
                masked_scores = np.where(self.eliminated, np.inf, self.match_scores)
                leader = int(np.argmin(masked_scores))
            else:
                leader = int(np.argmin(self.match_scores))
            info["match_leader"] = leader
            info["match_winner"] = self._match_committed_winner
        info["episode_play_when_available_ratio"] = (
            float(self._episode_play_when_available) / float(self._episode_play_options_turns)
            if self._episode_play_options_turns > 0
            else 0.0
        )
        return info

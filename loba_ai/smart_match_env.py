from __future__ import annotations

import numpy as np
from gymnasium import spaces

from loba_ai.cards import Card
from loba_ai.match_env import MatchLobaEnv


class SmartMatchLobaEnv(MatchLobaEnv):
    """Match env with round-level seen-card memory features."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._max_token_counts = np.full(53, float(self.rules.num_decks), dtype=np.float32)
        self._max_token_counts[52] = float(self.rules.num_jokers)
        self._total_cards = float((self.rules.num_decks * 52) + self.rules.num_jokers)
        self._seen_card_uids: set[tuple[int, str | None, int, bool]] = set()
        self._seen_token_counts = np.zeros(53, dtype=np.float32)

        base_dim = int(self.observation_space.shape[0])
        smart_dim = base_dim + 53 + 53 + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(smart_dim,), dtype=np.float32)

    def _card_uid(self, c: Card) -> tuple[int, str | None, int, bool]:
        return (c.rank, c.suit, c.deck_id, c.is_joker)

    def _reset_seen_tracker(self) -> None:
        self._seen_card_uids = set()
        self._seen_token_counts = np.zeros(53, dtype=np.float32)

    def _mark_seen(self, cards: list[Card] | tuple[Card, ...]) -> None:
        for c in cards:
            uid = self._card_uid(c)
            if uid in self._seen_card_uids:
                continue
            self._seen_card_uids.add(uid)
            self._seen_token_counts[c.token] += 1.0

    def _update_seen_from_state(self) -> None:
        s = self.engine.state
        self._mark_seen(s.players[0].hand)
        self._mark_seen(s.discard_pile)
        for meld in s.melds_on_table:
            self._mark_seen(meld)

    def _smart_features(self) -> np.ndarray:
        seen_ratio = np.clip(self._seen_token_counts / np.maximum(self._max_token_counts, 1.0), 0.0, 1.0)
        remaining_ratio = np.clip((self._max_token_counts - self._seen_token_counts) / np.maximum(self._max_token_counts, 1.0), 0.0, 1.0)
        seen_total = np.array([min(1.0, float(np.sum(self._seen_token_counts)) / self._total_cards)], dtype=np.float32)
        return np.concatenate([seen_ratio.astype(np.float32), remaining_ratio.astype(np.float32), seen_total])

    def _obs(self) -> np.ndarray:
        self._update_seen_from_state()
        base = super()._obs()
        return np.concatenate([base, self._smart_features()]).astype(np.float32)

    def _finish_round(self) -> tuple[float, bool, dict]:
        reward, done, info = super()._finish_round()
        if not done:
            # Reset card counting memory at the start of each new round.
            self._reset_seen_tracker()
            self._update_seen_from_state()
        return reward, done, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        _, info = super().reset(seed=seed, options=options)
        self._reset_seen_tracker()
        self._update_seen_from_state()
        return self._obs(), info

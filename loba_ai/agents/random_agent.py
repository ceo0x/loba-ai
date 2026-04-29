from __future__ import annotations

import numpy as np


class RandomAgent:
    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    def act(self, action_mask: np.ndarray) -> int:
        valid = np.flatnonzero(action_mask)
        return int(self.rng.choice(valid))

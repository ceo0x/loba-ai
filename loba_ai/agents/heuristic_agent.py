from __future__ import annotations

import numpy as np


class HeuristicAgent:
    """Simple policy: prefer meld actions, otherwise random valid."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    def act(self, action_mask: np.ndarray, phase: str) -> int:
        valid = np.flatnonzero(action_mask)
        if phase == "meld":
            meld_actions = [a for a in valid if 3 <= a < 3 + 24]
            if meld_actions:
                return int(meld_actions[0])
        if phase == "draw" and action_mask[1]:
            return 1
        return int(self.rng.choice(valid))

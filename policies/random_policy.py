"""Random policy — useful as a baseline / sanity check."""

import numpy as np
from .base import BasePolicy


class RandomPolicy(BasePolicy):
    """Outputs random actions uniformly in [-1, 1]."""

    def select_action(self, observation: dict) -> np.ndarray:
        return np.random.uniform(-1, 1, size=7).astype(np.float32)

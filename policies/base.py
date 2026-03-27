"""Abstract base class for all policies."""

from abc import ABC, abstractmethod
import numpy as np


class BasePolicy(ABC):
    """Base class for SimEngine policies.

    All policies receive observations and return actions:
        observation: dict with keys
            "state":  np.ndarray (7,) — joint angles in degrees + gripper 0-100
            "images": dict of {"webcam": (H,W,3), "arm_cam": (H,W,3)} uint8
            "task":   str — language task description
        action: np.ndarray (7,) in [-1, 1] — normalized joint targets + gripper
    """

    @abstractmethod
    def select_action(self, observation: dict) -> np.ndarray:
        """Given an observation, return an action."""
        pass

    def reset(self):
        """Called at the start of each episode. Override if needed."""
        pass

    def save(self, path: str):
        """Save policy weights. Override for trainable policies."""
        raise NotImplementedError

    def load(self, path: str):
        """Load policy weights. Override for trainable policies."""
        raise NotImplementedError

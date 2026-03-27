"""Simple MLP policy — fully implemented example for students.

This is a state-only policy (no images) that maps joint state -> action
using a 3-layer MLP. Students can use this as a reference when building
their own ACT/Diffusion/SmolVLA implementations.
"""

import numpy as np
import torch
import torch.nn as nn

from .base import BasePolicy


class MLPNetwork(nn.Module):
    """3-layer MLP: state(7) -> 256 -> 256 -> action(7)"""

    def __init__(self, state_dim=7, action_dim=7, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # actions in [-1, 1]
        )

    def forward(self, state):
        return self.net(state)


class MLPPolicy(BasePolicy):
    """Trainable MLP policy for behavior cloning.

    Input: state (7,) — joint angles + gripper
    Output: action (7,) — normalized joint targets + gripper
    """

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = MLPNetwork().to(self.device)
        self.state_mean = np.zeros(7, dtype=np.float32)
        self.state_std = np.ones(7, dtype=np.float32)
        self.action_mean = np.zeros(7, dtype=np.float32)
        self.action_std = np.ones(7, dtype=np.float32)

    def select_action(self, observation: dict) -> np.ndarray:
        state = observation["state"]
        # Normalize state
        state_norm = (state - self.state_mean) / self.state_std
        state_t = torch.from_numpy(state_norm).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_norm = self.model(state_t).cpu().numpy()[0]

        # Unnormalize action
        action = action_norm * self.action_std + self.action_mean
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def set_normalization(self, stats: dict):
        """Set normalization statistics from dataset."""
        self.state_mean = stats["state_mean"]
        self.state_std = stats["state_std"]
        self.action_mean = stats["action_mean"]
        self.action_std = stats["action_std"]

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.state_mean = ckpt["state_mean"]
        self.state_std = ckpt["state_std"]
        self.action_mean = ckpt["action_mean"]
        self.action_std = ckpt["action_std"]

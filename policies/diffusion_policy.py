"""Diffusion Policy skeleton.

Students: Implement the denoising diffusion model for action prediction.

Reference paper: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
Key ideas:
  - Model actions as samples from a learned diffusion process
  - Start from Gaussian noise, iteratively denoise to get action sequence
  - Condition on visual observations + state via FiLM or cross-attention
  - Produces diverse, multi-modal action distributions (unlike MSE-based policies)

Architecture overview:
  Input: state (7,) + webcam (3,H,W) + arm_cam (3,H,W)
  -> Visual encoder (ResNet18) extracts features
  -> Condition vector = concat(image_features, state)
  -> Noise predictor (1D U-Net or Transformer):
     takes noisy action chunk + condition + timestep -> predicts noise
  -> DDPM/DDIM sampling: iteratively denoise from random noise to actions
  -> Output: action chunk (horizon, 7)

Training:
  1. Sample random timestep t
  2. Add noise to ground-truth actions: a_t = sqrt(alpha_t) * a_0 + sqrt(1-alpha_t) * eps
  3. Predict noise: eps_pred = model(a_t, condition, t)
  4. Loss = MSE(eps_pred, eps)
"""

import numpy as np
import torch
import torch.nn as nn

from .base import BasePolicy


class DiffusionPolicy(BasePolicy):
    """Diffusion-based visuomotor policy.

    TODO: Implement the following:
    1. __init__: Create noise prediction network, visual encoder, noise schedule
    2. forward: Training forward pass (predict noise given noisy actions + condition)
    3. select_action: Run DDPM/DDIM denoising loop to sample actions
    4. compute_loss: Simple MSE between predicted and actual noise
    """

    def __init__(self, action_horizon=16, n_diffusion_steps=50,
                 hidden_dim=256, device="cpu"):
        self.action_horizon = action_horizon
        self.n_diffusion_steps = n_diffusion_steps
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self._action_queue = []

        # TODO: Initialize components
        # self.visual_encoder = ...    # ResNet18
        # self.noise_predictor = ...   # U-Net1D or Transformer
        # self.noise_schedule = ...    # Beta schedule: linear or cosine
        #
        # Hint: Use the diffusers library for the noise schedule:
        #   from diffusers import DDPMScheduler
        #   self.scheduler = DDPMScheduler(num_train_timesteps=n_diffusion_steps)
        raise NotImplementedError(
            "Diffusion policy not yet implemented. See docstring for architecture.\n"
            "Hint: The `diffusers` library (already installed) provides DDPMScheduler\n"
            "and other utilities that simplify implementation."
        )

    def select_action(self, observation: dict) -> np.ndarray:
        # TODO: Implement DDPM inference
        # 1. Encode observation to condition vector
        # 2. Start from random noise: a_T ~ N(0, I)
        # 3. For t = T, T-1, ..., 1:
        #    eps_pred = self.noise_predictor(a_t, condition, t)
        #    a_{t-1} = denoise_step(a_t, eps_pred, t)
        # 4. Return a_0[0] (first action) and queue the rest
        raise NotImplementedError

    def reset(self):
        self._action_queue = []

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError

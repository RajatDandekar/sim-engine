"""ACT (Action Chunking with Transformers) policy skeleton.

Students: Implement the model architecture and training logic below.

Reference paper: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
Key ideas:
  - CVAE (Conditional VAE) encodes action chunks during training
  - Transformer decoder predicts action sequences from visual + state input
  - Action chunking: predict N future actions at once for temporal consistency
  - Visual backbone: ResNet18 encodes camera images

Architecture overview:
  Input: state (7,) + webcam (3,H,W) + arm_cam (3,H,W) + task (str)
  -> ResNet18 encodes each image to a feature vector
  -> Concatenate [image_features, state] as transformer input tokens
  -> CVAE encoder (training only): encodes ground-truth action chunk -> z
  -> Transformer decoder: cross-attends to input tokens, conditioned on z
  -> Output: action chunk (chunk_size, 7) — sequence of future actions
"""

import numpy as np
import torch
import torch.nn as nn

from .base import BasePolicy


class ACTPolicy(BasePolicy):
    """Action Chunking with Transformers.

    TODO: Implement the following:
    1. __init__: Create ResNet18 visual encoder, Transformer encoder/decoder, CVAE
    2. encode_images: Process webcam + arm_cam through ResNet18
    3. forward: Full forward pass for training (with CVAE encoder)
    4. select_action: Inference — decode actions from visual + state input
    5. compute_loss: MSE on actions + KL divergence on CVAE latent
    """

    def __init__(self, chunk_size=20, hidden_dim=256, device="cpu"):
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self._action_queue = []  # buffered predicted actions

        # TODO: Initialize model components
        # self.visual_encoder = ...  # ResNet18, outputs (batch, hidden_dim) per image
        # self.state_encoder = ...   # Linear(7, hidden_dim)
        # self.transformer = ...     # nn.TransformerDecoder
        # self.cvae_encoder = ...    # For training: encodes action chunk -> mean, logvar
        # self.action_head = ...     # Linear(hidden_dim, 7)
        raise NotImplementedError(
            "ACT policy not yet implemented. See docstring for architecture details.\n"
            "Hint: Start by implementing encode_images() and a simple forward pass,\n"
            "then add the CVAE and action chunking."
        )

    def select_action(self, observation: dict) -> np.ndarray:
        # TODO: Implement inference
        # 1. If action queue is empty, run model to predict chunk_size actions
        # 2. Pop and return the first action from the queue
        #
        # if not self._action_queue:
        #     state = observation["state"]
        #     images = observation["images"]
        #     chunk = self._predict_chunk(state, images)
        #     self._action_queue = list(chunk)
        # return self._action_queue.pop(0)
        raise NotImplementedError

    def reset(self):
        self._action_queue = []

    def save(self, path: str):
        # TODO: torch.save(self.model.state_dict(), path)
        raise NotImplementedError

    def load(self, path: str):
        # TODO: self.model.load_state_dict(torch.load(path))
        raise NotImplementedError

"""SmolVLA (Small Vision-Language-Action) policy skeleton.

Students: Implement a language-conditioned visuomotor policy.

Reference: SmolVLA uses a pretrained Vision-Language Model (VLM) as the backbone,
with a lightweight action expert decoder on top.

Key ideas:
  - VLM backbone processes images + language task description jointly
  - The VLM's internal representations capture semantic understanding
  - A small action expert (MLP or diffusion head) decodes VLM features to actions
  - Language conditioning: same model handles "pick the red box" vs "pick the blue box"
  - Action chunking: predict 50 steps at once for smooth execution

Architecture overview:
  Input: webcam (3,H,W) + arm_cam (3,H,W) + state (7,) + task (str)
  -> Tokenize task string
  -> VLM processes: [image_tokens, text_tokens] -> hidden states
  -> Extract VLM features (last hidden state or pooled output)
  -> Action expert: VLM_features + state -> action chunk (50, 7)
  -> Use action queue for smooth execution

For this simulation, you can use a simplified version:
  - Replace VLM with a CNN (ResNet18) + text encoder (nn.Embedding or sentence-transformers)
  - Or use HuggingFace transformers to load a small VLM

Training:
  - Freeze VLM backbone (or fine-tune with low LR)
  - Train action expert with MSE loss on action chunks
  - Normalize state/action with dataset statistics
"""

import numpy as np
import torch
import torch.nn as nn

from .base import BasePolicy


class SmolVLAPolicy(BasePolicy):
    """Language-conditioned visuomotor policy.

    TODO: Implement the following:
    1. __init__: Create visual encoder, text encoder, action expert
    2. encode_observation: Process images + text into a feature vector
    3. forward: Full forward pass for training
    4. select_action: Inference with action chunking queue
    5. compute_loss: MSE on action chunks

    Simplified version (recommended to start):
      - ResNet18 for images
      - Simple learned embedding for a fixed set of task strings
      - MLP action expert: concat(image_feat, text_feat, state) -> actions
    """

    def __init__(self, chunk_size=50, hidden_dim=512, device="cpu"):
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self._action_queue = []

        # TODO: Initialize components
        # Option A (simple):
        #   self.image_encoder = torchvision.models.resnet18(pretrained=True)
        #   self.text_encoder = nn.Embedding(num_tasks, hidden_dim)
        #   self.action_expert = nn.Sequential(
        #       nn.Linear(512 + hidden_dim + 7, 512),
        #       nn.ReLU(),
        #       nn.Linear(512, chunk_size * 7),
        #   )
        #
        # Option B (VLM-based):
        #   from transformers import AutoModel, AutoTokenizer
        #   self.vlm = AutoModel.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
        #   self.action_expert = nn.Linear(vlm_dim, chunk_size * 7)
        raise NotImplementedError(
            "SmolVLA policy not yet implemented. See docstring for architecture.\n"
            "Hint: Start with Option A (simple) to get a working pipeline,\n"
            "then upgrade to a real VLM backbone."
        )

    def select_action(self, observation: dict) -> np.ndarray:
        # TODO: Implement with action chunking
        # if not self._action_queue:
        #     features = self.encode_observation(observation)
        #     chunk = self.action_expert(features)  # (chunk_size, 7)
        #     self._action_queue = list(chunk)
        # return self._action_queue.pop(0)
        raise NotImplementedError

    def reset(self):
        self._action_queue = []

    def save(self, path: str):
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError

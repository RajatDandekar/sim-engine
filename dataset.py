"""HDF5 dataset recorder and PyTorch-compatible loader."""

import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from . import config


class DatasetRecorder:
    """Records episodes to HDF5 files."""

    def __init__(self, save_dir=None):
        self.save_dir = save_dir or config.DEFAULT_DATASET_DIR
        os.makedirs(self.save_dir, exist_ok=True)
        self._buffers = None
        self._task = None
        self._episode_idx = self._count_existing()

    def _count_existing(self):
        if not os.path.exists(self.save_dir):
            return 0
        return len([f for f in os.listdir(self.save_dir) if f.endswith(".hdf5")])

    def start_episode(self, task: str = None):
        self._task = task or config.DEFAULT_TASK
        self._buffers = {
            "states": [],
            "images_webcam": [],
            "images_arm_cam": [],
            "actions": [],
        }

    def record_step(self, obs: dict, action: np.ndarray):
        assert self._buffers is not None, "Call start_episode() first"
        self._buffers["states"].append(obs["state"].copy())
        self._buffers["actions"].append(action.copy())
        if "images" in obs:
            self._buffers["images_webcam"].append(obs["images"]["webcam"].copy())
            self._buffers["images_arm_cam"].append(obs["images"]["arm_cam"].copy())

    def end_episode(self, success: bool = False):
        assert self._buffers is not None, "Call start_episode() first"

        fname = f"episode_{self._episode_idx:04d}.hdf5"
        fpath = os.path.join(self.save_dir, fname)

        n_steps = len(self._buffers["states"])
        with h5py.File(fpath, "w") as f:
            obs_g = f.create_group("observations")
            obs_g.create_dataset("state", data=np.array(self._buffers["states"], dtype=np.float32))

            if self._buffers["images_webcam"]:
                img_g = obs_g.create_group("images")
                img_g.create_dataset(
                    "webcam",
                    data=np.array(self._buffers["images_webcam"], dtype=np.uint8),
                    compression="gzip", compression_opts=4,
                )
                img_g.create_dataset(
                    "arm_cam",
                    data=np.array(self._buffers["images_arm_cam"], dtype=np.uint8),
                    compression="gzip", compression_opts=4,
                )

            f.create_dataset("actions", data=np.array(self._buffers["actions"], dtype=np.float32))
            f.attrs["task"] = self._task
            f.attrs["fps"] = config.CONTROL_FREQ
            f.attrs["success"] = success
            f.attrs["num_steps"] = n_steps

        print(f"  Saved {fname} ({n_steps} steps, success={success})")
        self._episode_idx += 1
        self._buffers = None
        return fpath


class SimDataset(Dataset):
    """PyTorch Dataset that loads episodes from HDF5 files.

    Returns (state, action) pairs for training state-based policies,
    or (state, images, action) if load_images=True.
    """

    def __init__(self, data_dir=None, load_images=False):
        self.data_dir = data_dir or config.DEFAULT_DATASET_DIR
        self.load_images = load_images

        # Scan for episode files
        self.episode_files = sorted([
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if f.endswith(".hdf5")
        ])
        assert len(self.episode_files) > 0, f"No episodes found in {self.data_dir}"

        # Build index: (episode_idx, step_idx) for each global index
        self._index = []
        self._states = []
        self._actions = []
        self._images_webcam = []
        self._images_arm_cam = []
        self._tasks = []

        for ep_idx, fpath in enumerate(self.episode_files):
            with h5py.File(fpath, "r") as f:
                states = f["observations/state"][:]
                actions = f["actions"][:]
                task = f.attrs.get("task", config.DEFAULT_TASK)

                self._states.append(states)
                self._actions.append(actions)
                self._tasks.append(task)

                if load_images and "observations/images/webcam" in f:
                    self._images_webcam.append(f["observations/images/webcam"][:])
                    self._images_arm_cam.append(f["observations/images/arm_cam"][:])

                for step in range(len(states)):
                    self._index.append((ep_idx, step))

        # Concatenate for easy normalization
        all_states = np.concatenate(self._states, axis=0)
        all_actions = np.concatenate(self._actions, axis=0)

        self.state_mean = all_states.mean(axis=0).astype(np.float32)
        self.state_std = all_states.std(axis=0).astype(np.float32) + 1e-6
        self.action_mean = all_actions.mean(axis=0).astype(np.float32)
        self.action_std = all_actions.std(axis=0).astype(np.float32) + 1e-6

        print(f"Loaded {len(self.episode_files)} episodes, {len(self._index)} total steps")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ep_idx, step_idx = self._index[idx]

        state = self._states[ep_idx][step_idx].astype(np.float32)
        action = self._actions[ep_idx][step_idx].astype(np.float32)

        # Normalize
        state_norm = (state - self.state_mean) / self.state_std
        action_norm = (action - self.action_mean) / self.action_std

        result = {
            "state": torch.from_numpy(state_norm),
            "action": torch.from_numpy(action_norm),
            "state_raw": torch.from_numpy(state),
            "action_raw": torch.from_numpy(action),
        }

        if self.load_images and self._images_webcam:
            webcam = self._images_webcam[ep_idx][step_idx]
            arm_cam = self._images_arm_cam[ep_idx][step_idx]
            # HWC uint8 -> CHW float32 [0, 1]
            result["webcam"] = torch.from_numpy(webcam).permute(2, 0, 1).float() / 255.0
            result["arm_cam"] = torch.from_numpy(arm_cam).permute(2, 0, 1).float() / 255.0

        return result

    def get_normalization_stats(self):
        """Return normalization statistics for use during inference."""
        return {
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
        }

# SimEngine - SO-101 Pick & Place Simulator

A MuJoCo-based simulation engine for learning robotic manipulation policies. Built for the robotics bootcamp.

Simulate a SO-101 robot arm performing pick-and-place tasks. Record demonstrations, implement your own policy (ACT, Diffusion, SmolVLA, or anything custom), train it, and evaluate -- all from a single UI.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [The Robot](#the-robot)
3. [The UI](#the-ui)
4. [Step-by-Step Guide](#step-by-step-guide)
5. [Implementing Your Own Policy](#implementing-your-own-policy)
6. [Understanding the Code](#understanding-the-code)
7. [How the Physics Works](#how-the-physics-works)
8. [Tips for Better Policies](#tips-for-better-policies)
9. [CLI Reference](#cli-reference)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/RajatDandekar/sim-engine.git
cd sim-engine
pip install -e .
```

If you're using a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows
pip install -e .
```

### 2. Launch

```bash
python -m sim_engine
```

That's it. The UI will open.

### 3. First run (5 minutes)

1. Press **5** to watch a demo of the scripted policy picking and placing
2. Press **1** to record 10 demonstration episodes
3. Press **3** to train the MLP policy on your demos
4. Press **4** to evaluate and watch the MLP try to replicate the task

---

## The Robot

**SO-101** is a 6-DOF manipulator arm with a parallel jaw gripper, modeled after the real SO-101 hardware used in the bootcamp.

```
                    [wrist_roll]
                        |
                   [wrist_flex]
                        |
                   [elbow_flex]
                       /
              [shoulder_lift]
                    |
             [shoulder_pan]
                    |
              [base on table]
```

### Specifications

| Property | Value |
|----------|-------|
| Arm joints | 5 (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll) |
| Gripper | Parallel jaw, 2 coupled fingers |
| Total DOF | 7 (5 arm + 2 gripper) |
| Control | Position control at 30 Hz |
| Cameras | 2 (overhead webcam + wrist-mounted) |

### Joint Order (important!)

The joint order matches the real SO-101. This matters when you build your policy:

```
Index 0: shoulder_pan   (base rotation)
Index 1: shoulder_lift  (shoulder up/down)
Index 2: elbow_flex     (elbow bend)
Index 3: wrist_flex     (wrist up/down)
Index 4: wrist_roll     (wrist rotation)
Index 5: gripper        (-1 = close, +1 = open)
Index 6: gripper        (same as index 5, coupled)
```

---

## The UI

Launch with `python -m sim_engine` (or `python -m sim_engine app`).

### Main Menu

| Key | Screen | Description |
|-----|--------|-------------|
| **1** | Record | Collect demonstration episodes using the scripted policy |
| **2** | Policies | View available policies, create new ones, open in editor |
| **3** | Train | Train any implemented policy on recorded demonstrations |
| **4** | Evaluate | Run a trained policy and watch it perform live |
| **5** | Demo | Watch the scripted baseline pick and place |
| **q** | Quit | Exit the application |

### Record Screen
- Automatically runs the scripted policy for 10 episodes
- Each episode is saved as an HDF5 file in `datasets/sim_demos/`
- Shows live rendering with wrist camera picture-in-picture
- Red blinking dot indicates recording is active

### Policies Screen
- Lists all policies and their status (implemented / skeleton)
- Press a policy's key to open it in your code editor (VS Code or default)
- Press **n** to create a brand new policy from a template
- Press **o** to open the policies folder

### Train Screen
- Lists all implemented policies
- Select one to train (100 epochs, MSE loss on actions)
- Shows live loss curve during training
- Press **ESC** to stop training early
- Checkpoints saved to `outputs/sim_engine/<policy_name>/`

### Evaluate Screen
- Lists scripted baseline, random baseline, and all trained policies
- Select one to watch it run for 10 episodes
- Press **SPACE** before each episode to start
- Shows success rate and average reward at the end

---

## Step-by-Step Guide

### Step 1: Understand the task

The robot needs to:
1. Move the gripper above the red box
2. Lower down to the box
3. Close the gripper (action[5] = -1)
4. Lift the box
5. Move over the blue bowl
6. Lower into the bowl
7. Open the gripper (action[5] = +1)
8. Return to home position

### Step 2: Record demonstrations

From the UI, press **1** to record. The scripted policy will perform the task automatically. Record at least **50 episodes** for good results.

Each episode saves:
- Joint states at every timestep: shape `(T, 7)`
- Actions at every timestep: shape `(T, 7)`
- Camera images from both cameras: `(T, 480, 640, 3)`
- Task description string
- Success flag

Data is stored in `datasets/sim_demos/episode_XXXX.hdf5`.

### Step 3: Implement your policy

From the UI, press **2** then press the key for the policy you want to edit. It opens in your code editor.

Your policy must implement one key method:

```python
def select_action(self, observation: dict) -> np.ndarray:
```

**Input** - `observation` dict:
```python
{
    "state":  np.ndarray,  # shape (7,) - joint angles in degrees + gripper (0-100)
    "images": {
        "webcam":  np.ndarray,  # shape (480, 640, 3) uint8 RGB
        "arm_cam": np.ndarray,  # shape (480, 640, 3) uint8 RGB
    },
    "task": str,  # "Pick up the box and place it in the bowl"
}
```

**Output** - action array:
```python
np.ndarray  # shape (7,) in [-1, 1]
# [0]: shoulder_pan target (normalized)
# [1]: shoulder_lift target
# [2]: elbow_flex target
# [3]: wrist_flex target
# [4]: wrist_roll target
# [5]: gripper command (-1 = close, +1 = open)
# [6]: gripper command (same as [5])
```

### Step 4: Train

From the UI, press **3** and select your policy. Training runs for 100 epochs with:
- Batch size: 64
- Learning rate: 3e-4
- Loss: MSE between predicted and recorded actions
- Optimizer: Adam

Watch the loss curve drop in real-time. The best checkpoint is saved automatically.

### Step 5: Evaluate

From the UI, press **4** and select your trained policy. Watch it attempt the task for 10 episodes. Compare success rates between different architectures.

---

## Implementing Your Own Policy

### Option A: Edit a provided skeleton

Detailed skeletons with architecture documentation are provided for three popular approaches:

| File | Architecture | Key Ideas |
|------|-------------|-----------|
| `policies/act_policy.py` | ACT | CVAE + Transformer decoder, action chunking |
| `policies/diffusion_policy.py` | Diffusion Policy | DDPM denoising, multi-modal actions |
| `policies/smolvla_policy.py` | SmolVLA | VLM backbone, language-conditioned |

Each skeleton contains:
- Detailed docstrings explaining the architecture
- Pseudocode for the forward pass
- Hints about which libraries to use
- TODO markers where you need to add code

### Option B: Create a new policy from the UI

Press **2** (Policies) then **n** (New). Select a name and a working template is generated with:
- A basic neural network (nn.Module)
- `select_action()` already wired up with normalization
- `save()` and `load()` methods for checkpointing
- Ready to train immediately

You can then modify the network architecture to experiment.

### Option C: Create manually

Create `policies/my_policy.py`:

```python
import numpy as np
import torch
import torch.nn as nn
from .base import BasePolicy


class MyNetwork(nn.Module):
    """Define your neural network here."""

    def __init__(self, state_dim=7, action_dim=7, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, state):
        return self.net(state)


class MyPolicy(BasePolicy):

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = MyNetwork().to(self.device)
        self.state_mean = np.zeros(7, dtype=np.float32)
        self.state_std = np.ones(7, dtype=np.float32)
        self.action_mean = np.zeros(7, dtype=np.float32)
        self.action_std = np.ones(7, dtype=np.float32)

    def select_action(self, observation: dict) -> np.ndarray:
        state = observation["state"]
        # Normalize
        state_norm = (state - self.state_mean) / self.state_std
        state_t = torch.from_numpy(state_norm).float().unsqueeze(0).to(self.device)
        # Predict
        with torch.no_grad():
            action_norm = self.model(state_t).cpu().numpy()[0]
        # Unnormalize
        action = action_norm * self.action_std + self.action_mean
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def set_normalization(self, stats: dict):
        self.state_mean = stats["state_mean"]
        self.state_std = stats["state_std"]
        self.action_mean = stats["action_mean"]
        self.action_std = stats["action_std"]

    def reset(self):
        pass

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
```

The training pipeline automatically:
- Loads your `model` attribute for `optimizer = Adam(policy.model.parameters())`
- Calls `set_normalization()` with dataset statistics
- Saves checkpoints via your `save()` method

---

## Understanding the Code

### File Structure

```
sim_engine/
    __main__.py                # Entry point
    app.py                     # Full UI application
    config.py                  # Constants (joint names, camera settings, paths)
    env.py                     # SO101PickPlaceEnv (Gymnasium environment)
    renderer.py                # MuJoCo offscreen camera rendering
    dataset.py                 # HDF5 recorder + PyTorch Dataset loader
    ik_controller.py           # Jacobian-based inverse kinematics
    viewer.py                  # Simple demo viewer (alternative to app.py)
    record.py                  # CLI: record episodes
    train.py                   # CLI: train policies
    evaluate.py                # CLI: evaluate policies

    assets/
        so101_scene.xml        # MuJoCo scene definition (robot + table + objects)

    policies/
        base.py                # BasePolicy abstract class
        random_policy.py       # Random actions (baseline)
        scripted_policy.py     # Waypoint IK controller (generates demos)
        mlp_policy.py          # Simple MLP (fully working example)
        act_policy.py          # ACT skeleton with TODOs
        diffusion_policy.py    # Diffusion Policy skeleton with TODOs
        smolvla_policy.py      # SmolVLA skeleton with TODOs
```

### Key Classes

**`SO101PickPlaceEnv`** (`env.py`) - The Gymnasium environment.
- `reset()` - Resets arm to home, randomizes box position
- `step(action)` - Applies action, steps physics, returns (obs, reward, terminated, truncated, info)
- `info["success"]` - True when box is inside the bowl

**`DatasetRecorder`** (`dataset.py`) - Saves episodes to HDF5.
- `start_episode(task)` - Begin recording
- `record_step(obs, action)` - Store one timestep
- `end_episode(success)` - Save to disk

**`SimDataset`** (`dataset.py`) - PyTorch Dataset for training.
- Loads all HDF5 files from a directory
- Computes normalization statistics (mean/std)
- Returns `{"state": ..., "action": ...}` tensors

**`BasePolicy`** (`policies/base.py`) - Interface all policies must implement.
- `select_action(observation) -> action` - Required
- `reset()` - Called at episode start
- `save(path)` / `load(path)` - For checkpointing

---

## How the Physics Works

### MuJoCo Simulation

The scene (`assets/so101_scene.xml`) defines:
- A table with a wood texture
- The SO-101 arm mounted on the table (position-controlled actuators)
- A red box (free body with a freejoint)
- A blue bowl (static body made of box geoms)
- Two cameras and three lights

Physics runs at 500 Hz internally (timestep = 0.002s), but the policy runs at 30 Hz. Each `env.step()` takes ~16 physics substeps.

### Grasping (Kinematic Snap-Grasp)

Real friction-based grasping in simulation is unreliable. Instead, SimEngine uses **kinematic snap-grasp**, the same approach used by Robosuite, MetaWorld, and other standard benchmarks:

1. **Grasp triggers** when ALL of these are true:
   - Gripper action < -0.5 (close command)
   - Finger joint position < 0.018 (fingers actually closing)
   - End-effector is within 6cm of the box center

2. **While grasped**: The box position is set to track the end-effector every step (kinematic attachment). The box velocity is zeroed to prevent it flying off on release.

3. **Release triggers** when gripper action > 0.5 (open command). The box is then free to fall under gravity.

This means your policy needs to learn a specific sequence:
- Get the end-effector close to the box (within 6cm)
- Output a strong close signal (action[5] = -1.0)
- Keep it closed while transporting
- Output a strong open signal (action[5] = +1.0) when over the bowl

### Reward Function

```
reward = -0.1 * distance(EE, box)                          # approach shaping
       + 1.0 * (box is lifted above table)                  # grasp bonus
       - 0.1 * distance(box, bowl)  [only if box lifted]    # transport shaping
       + 5.0 * (box is inside bowl)                          # placement bonus
```

An episode terminates with `success=True` when the box center is inside the bowl radius and between the bowl base and rim height.

---

## Tips for Better Policies

### Why the simple MLP fails

The MLP maps state -> action independently at each timestep. But pick-and-place requires **temporal reasoning**: first approach, then grasp, then lift, then place. A single feedforward pass can't capture this multi-phase behavior.

### What works better

1. **Action chunking** (ACT, SmolVLA): Predict 20-50 future actions at once. The policy is queried less often and produces smooth, temporally coherent trajectories.

2. **Diffusion**: Model the action distribution with a denoising process. Handles multi-modal demonstrations (different paths to the same goal).

3. **Recurrent / Transformer**: Add memory so the policy knows what phase it's in (approaching vs grasping vs transporting).

4. **Use images**: The state-only MLP ignores camera input. Vision-based policies (CNN + MLP, or ViT) can learn spatial relationships directly.

### Practical tips

- **Record 50+ episodes** minimum. 100 is better.
- **Start state-only**, then add images once it works.
- **Action chunking is the single biggest improvement** over step-by-step prediction.
- **Check your normalization**: the training pipeline normalizes state and action by mean/std. Make sure your `select_action` unnormalizes correctly.
- **Watch the loss curve**: if it plateaus early, try a larger model or lower learning rate.
- **Compare against scripted**: the scripted policy gets ~90% success. That's your target.

---

## CLI Reference

For advanced usage without the UI:

```bash
# Record demonstrations
python -m sim_engine record --mode scripted -n 50
python -m sim_engine record --mode keyboard          # manual teleoperation

# Train (any implemented policy)
python -m sim_engine train --policy mlp --epochs 200 --batch_size 64 --lr 3e-4
python -m sim_engine train --policy diffusion --epochs 200 --device cuda

# Evaluate
python -m sim_engine eval --policy scripted -n 10
python -m sim_engine eval --policy mlp --checkpoint outputs/sim_engine/mlp/best.pt -n 10
python -m sim_engine eval --policy diffusion --checkpoint outputs/sim_engine/diffusion/best.pt

# Simple viewer
python -m sim_engine view

# Full UI (default)
python -m sim_engine app
```

---

## Troubleshooting

### "No module named 'mujoco'"
```bash
pip install mujoco
```

### "No module named 'cv2'" or black window
```bash
pip install opencv-python    # NOT opencv-python-headless
```

### Training loss doesn't decrease
- Record more episodes (at least 50)
- Check that episodes are successful (the scripted policy should succeed ~90%)
- Try a lower learning rate: `--lr 1e-4`

### Policy moves randomly during evaluation
- Make sure you're loading the checkpoint: `--checkpoint outputs/sim_engine/mlp/best.pt`
- Check that the policy's `load()` method restores normalization stats

### OpenCV window doesn't appear
- On macOS, make sure you're not using `opencv-python-headless`
- On Linux, you may need: `sudo apt-get install python3-tk`

### "No episodes found"
- Record episodes first: press **1** in the UI, or run `python -m sim_engine record --mode scripted -n 10`
- Episodes are saved to `datasets/sim_demos/`

### Want to reset all data
```bash
rm -rf datasets/ outputs/    # delete all recorded data and trained models
```

---

## Requirements

- Python 3.10+
- macOS, Linux, or Windows
- No GPU required (CPU is fine, GPU speeds up training)
- ~500 MB disk space for MuJoCo + dependencies

### Core dependencies (installed automatically)

| Package | Purpose |
|---------|---------|
| mujoco | Physics simulation |
| gymnasium | Environment interface |
| torch | Neural networks and training |
| opencv-python | Rendering and UI display |
| numpy | Numerical computation |
| h5py | Dataset storage (HDF5) |

### Optional (for advanced policies)

```bash
pip install -e ".[dev]"    # installs diffusers, transformers, torchvision
```

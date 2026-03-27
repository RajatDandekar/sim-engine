# SimEngine - SO-101 Pick & Place Simulator

A MuJoCo-based simulation engine for learning robotic manipulation policies. Built for the robotics bootcamp.

Simulate a SO-101 robot arm performing pick-and-place tasks. Record demonstrations, implement your own policy (ACT, Diffusion, SmolVLA, or anything custom), train it, and evaluate -- all from a single UI.

## Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/sim-engine.git
cd sim-engine
pip install -e .
```

### 2. Launch

```bash
python -m sim_engine
```

This opens the SimEngine UI with 5 options:

| Key | Action | What it does |
|-----|--------|-------------|
| 1 | Record | Collect pick-and-place demos (auto or keyboard) |
| 2 | Policies | View, create, and edit policy architectures |
| 3 | Train | Train any implemented policy on recorded data |
| 4 | Evaluate | Run a policy and watch it perform |
| 5 | Demo | Watch the scripted baseline in action |

## The Robot

**SO-101** - 6-DOF manipulator arm with a parallel jaw gripper.

- **5 arm joints**: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll
- **2 finger joints**: coupled parallel gripper (open/close)
- **2 cameras**: overhead webcam + wrist-mounted camera
- **Task**: Pick up the red box, place it in the blue bowl

## Implement Your Own Policy

### Option A: Edit a skeleton

Skeletons with architecture docs and TODOs are provided for:
- `policies/act_policy.py` - Action Chunking with Transformers
- `policies/diffusion_policy.py` - Diffusion Policy
- `policies/smolvla_policy.py` - Vision-Language-Action model

### Option B: Create from scratch

From the UI: press **2** (Policies) then **n** (New policy). This generates a working template.

Or manually create `policies/my_policy.py`:

```python
import numpy as np
import torch
import torch.nn as nn
from .base import BasePolicy

class MyPolicy(BasePolicy):
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = MyNetwork().to(self.device)  # your nn.Module
        # normalization stats (set by training pipeline)
        self.state_mean = np.zeros(7, dtype=np.float32)
        self.state_std = np.ones(7, dtype=np.float32)
        self.action_mean = np.zeros(7, dtype=np.float32)
        self.action_std = np.ones(7, dtype=np.float32)

    def select_action(self, observation: dict) -> np.ndarray:
        """
        observation has:
            state:  (7,)  - joint angles in degrees + gripper 0-100
            images: dict  - webcam (480,640,3) + arm_cam (480,640,3)
            task:   str   - "Pick up the box and place it in the bowl"

        Return: (7,) action in [-1, 1]
            [0-4] = arm joint targets (normalized)
            [5-6] = gripper (-1 = close, +1 = open)
        """
        # Your inference code here
        ...

    def set_normalization(self, stats):
        self.state_mean = stats["state_mean"]
        self.state_std = stats["state_std"]
        self.action_mean = stats["action_mean"]
        self.action_std = stats["action_std"]

    def save(self, path):
        torch.save({"model": self.model.state_dict(), ...}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
```

## Workflow

```
Record demos  -->  Implement policy  -->  Train  -->  Evaluate
    [1]               [2]                 [3]         [4]
```

1. **Record 50+ episodes** using the scripted policy
2. **Implement** your policy architecture in the policies folder
3. **Train** from the UI (or CLI: `python -m sim_engine train --policy my`)
4. **Evaluate** and watch your policy try to pick and place

## How Grasping Works

The simulator uses **kinematic snap-grasp**:
- When the gripper closes (action < -0.5) near the box (< 6cm), the box locks to the gripper
- When the gripper opens (action > 0.5), the box releases
- This is the standard approach used by Robosuite, MetaWorld, etc.

Your policy needs to learn: approach box, close gripper, move to bowl, open gripper.

## CLI Commands

For advanced usage without the UI:

```bash
python -m sim_engine record --mode scripted -n 50    # Record 50 demos
python -m sim_engine train --policy mlp --epochs 200  # Train MLP
python -m sim_engine eval --policy mlp --checkpoint outputs/sim_engine/mlp/best.pt
python -m sim_engine view                              # Simple demo viewer
```

## File Structure

```
sim_engine/
    assets/so101_scene.xml     # MuJoCo scene (arm + table + box + bowl)
    config.py                  # Constants and paths
    env.py                     # Gymnasium environment
    renderer.py                # Camera rendering
    dataset.py                 # HDF5 dataset recorder/loader
    ik_controller.py           # Jacobian IK for scripted demos
    app.py                     # Full UI application
    policies/
        base.py                # BasePolicy interface
        random_policy.py       # Random baseline
        scripted_policy.py     # Waypoint IK (generates demos)
        mlp_policy.py          # Simple MLP (working example)
        act_policy.py          # ACT skeleton
        diffusion_policy.py    # Diffusion skeleton
        smolvla_policy.py      # SmolVLA skeleton
    record.py                  # CLI recording
    train.py                   # CLI training
    evaluate.py                # CLI evaluation
    viewer.py                  # Simple viewer
```

## Requirements

- Python 3.10+
- macOS, Linux, or Windows
- No GPU required (CPU works, GPU speeds up training)

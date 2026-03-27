"""Central configuration for the SimEngine."""

import os
import numpy as np

# Joint names in correct order (matches real SO-101)
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

N_JOINTS = 6  # arm joints (excluding gripper)
N_ACTIONS = 7  # 6 arm joints + 1 gripper

# Joint limits in radians (used by MuJoCo actuators)
JOINT_LIMITS_RAD = {
    "shoulder_pan":  (-np.pi, np.pi),
    "shoulder_lift": (-np.pi / 2, np.pi / 2),
    "elbow_flex":    (-np.pi, 0.0),
    "wrist_flex":    (-np.pi / 2, np.pi / 2),
    "wrist_roll":    (-np.pi, np.pi),
    "gripper":       (0.0, 0.04),  # finger separation in meters
}

# Home position in radians (arm straight up, out of workspace)
HOME_QPOS_RAD = np.array([0.0, 0.0, -0.3, -0.3, 0.0, 0.04])

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_NAMES = ["webcam", "arm_cam"]

# Physics
CONTROL_FREQ = 30  # Hz
PHYSICS_TIMESTEP = 0.002  # seconds
PHYSICS_SUBSTEPS = int(1.0 / CONTROL_FREQ / PHYSICS_TIMESTEP)

# Episode
MAX_EPISODE_STEPS = 500  # ~17 seconds at 30 Hz

# Task
DEFAULT_TASK = "Pick up the box and place it in the bowl"

# Paths
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
SCENE_XML = os.path.join(ASSETS_DIR, "so101_scene.xml")
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets", "sim_demos")
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "sim_engine")

# Object positions (meters, relative to world origin at table center)
BOX_DEFAULT_POS = np.array([0.15, 0.0, 0.44])   # on table surface
BOWL_DEFAULT_POS = np.array([-0.15, 0.0, 0.41])  # on table surface
BOWL_RADIUS = 0.05
BOWL_HEIGHT = 0.04

# Python executable (use the venv)
PYTHON = os.path.join(
    os.path.dirname(__file__), "..", "lerobot", ".venv", "bin", "python3"
)

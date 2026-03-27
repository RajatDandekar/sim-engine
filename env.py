"""Gymnasium environment for SO-101 pick-and-place in MuJoCo."""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from . import config
from .renderer import OffscreenRenderer


class SO101PickPlaceEnv(gym.Env):
    """Simulated SO-101 arm performing pick-and-place.

    Observation:
        state: (7,) float32 — 6 joint angles (degrees) + gripper opening (0-100)
        images: dict of webcam and arm_cam RGB arrays (H, W, 3) uint8

    Action:
        (7,) float32 in [-1, 1] — normalized joint targets + gripper command
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": config.CONTROL_FREQ}

    def __init__(self, render_mode=None, render_images=True):
        super().__init__()
        self.render_mode = render_mode
        self.render_images = render_images

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(config.SCENE_XML)
        self.model.opt.timestep = config.PHYSICS_TIMESTEP
        self.data = mujoco.MjData(self.model)

        # Camera renderer
        if self.render_images:
            self._renderer = OffscreenRenderer(
                self.model, config.CAMERA_WIDTH, config.CAMERA_HEIGHT
            )
        else:
            self._renderer = None

        # Cache body/joint/actuator IDs
        self._arm_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in config.JOINT_NAMES[:5]  # 5 arm joints
        ]
        self._finger_left_jnt = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_left"
        )
        self._box_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "box"
        )
        self._bowl_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "bowl"
        )
        self._ee_site = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )

        # Actuator control ranges
        self._ctrl_ranges = self.model.actuator_ctrlrange.copy()

        # Snap-grasp state (kinematic attachment when gripper closes near box)
        self._grasped = False
        self._grasp_offset = np.zeros(3)  # box offset from EE when grasped
        self._grasp_threshold = 0.06  # EE-to-box distance to trigger grasp
        self._gripper_close_threshold = 0.018  # finger qpos below this = closed
        self._box_qpos_adr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        ]
        self._box_dof_adr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        ]

        # Remember initial state for reset
        mujoco.mj_forward(self.model, self.data)
        self._init_qpos = self.data.qpos.copy()
        self._init_qvel = self.data.qvel.copy()

        # Spaces
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)

        obs_spaces = {"state": spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)}
        if self.render_images:
            obs_spaces["images"] = spaces.Dict({
                name: spaces.Box(0, 255, shape=(config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
                for name in config.CAMERA_NAMES
            })
        self.observation_space = spaces.Dict(obs_spaces)

        self._step_count = 0
        self._viewer = None

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset state
        self.data.qpos[:] = self._init_qpos
        self.data.qvel[:] = self._init_qvel
        self.data.ctrl[:] = 0.0

        # Set arm to home position
        for i, jname in enumerate(config.JOINT_NAMES[:5]):
            jid = self._arm_joint_ids[i]
            qadr = self.model.jnt_qposadr[jid]
            self.data.qpos[qadr] = config.HOME_QPOS_RAD[i]

        # Open gripper
        fl_qadr = self.model.jnt_qposadr[self._finger_left_jnt]
        self.data.qpos[fl_qadr] = 0.04  # fully open

        # Randomize box position slightly
        box_qpos_adr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        ]
        self.data.qpos[box_qpos_adr:box_qpos_adr + 3] = config.BOX_DEFAULT_POS + self.np_random.uniform(
            low=[-0.03, -0.03, 0.0], high=[0.03, 0.03, 0.0]
        )
        # Reset box orientation to identity quaternion
        self.data.qpos[box_qpos_adr + 3:box_qpos_adr + 7] = [1, 0, 0, 0]
        # Zero box velocity
        box_dof_adr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "box_joint")
        ]
        self.data.qvel[box_dof_adr:box_dof_adr + 6] = 0.0

        # Reset grasp state
        self._grasped = False
        self._grasp_offset = np.zeros(3)

        # Set actuator controls to match initial positions
        for i in range(5):
            jid = self._arm_joint_ids[i]
            qadr = self.model.jnt_qposadr[jid]
            self.data.ctrl[i] = self.data.qpos[qadr]
        self.data.ctrl[5] = 0.04  # left finger open
        self.data.ctrl[6] = 0.04  # right finger open

        # Settle physics
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self._step_count = 0
        obs = self._get_obs()
        return obs, self._get_info()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Scale action from [-1,1] to actuator control ranges
        # First 5: arm joints, 6th: gripper (controls both fingers)
        for i in range(5):
            lo, hi = self._ctrl_ranges[i]
            self.data.ctrl[i] = lo + (action[i] + 1.0) * 0.5 * (hi - lo)

        # Gripper: action[5] in [-1,1] -> finger separation [0.005, 0.04]
        # Min 0.005 prevents fingers from squeezing box out
        gripper_val = 0.005 + (action[5] + 1.0) * 0.5 * 0.035
        self.data.ctrl[5] = gripper_val  # left finger
        self.data.ctrl[6] = gripper_val  # right finger (coupled but set explicitly)

        # action[6] is the second gripper dim (ignored if coupled, or can be used for asymmetric grip)
        # For simplicity, we use action[5] for both fingers

        # Update snap-grasp (based on gripper action command)
        self._update_grasp(action[5])

        # Step physics
        for _ in range(config.PHYSICS_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        info = self._get_info()

        reward = self._compute_reward(info)
        terminated = info["success"]
        truncated = self._step_count >= config.MAX_EPISODE_STEPS

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array" and self._renderer is not None:
            return self._renderer.render(self.data, "webcam")
        elif self.render_mode == "human":
            self._render_human()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
        if self._viewer is not None:
            self._viewer.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_grasp(self, gripper_action):
        """Kinematic snap-grasp: attach box to EE when gripper closes near it.

        Grasp triggers when fingers are closed AND EE is near box.
        Release only when the gripper action commands open (> 0).
        """
        ee_pos = self.data.site_xpos[self._ee_site].copy()
        box_pos = self.data.xpos[self._box_body].copy()
        dist = np.linalg.norm(ee_pos - box_pos)

        fl_qadr = self.model.jnt_qposadr[self._finger_left_jnt]
        finger_qpos = self.data.qpos[fl_qadr]
        gripper_closing = gripper_action < -0.5  # action commands close
        gripper_opening = gripper_action > 0.5   # action commands open

        if not self._grasped and gripper_closing and finger_qpos < self._gripper_close_threshold and dist < self._grasp_threshold:
            self._grasp_offset = box_pos - ee_pos
            self._grasped = True
        elif self._grasped and gripper_opening:
            self._grasped = False

        if self._grasped:
            new_box_pos = ee_pos + self._grasp_offset
            self.data.qpos[self._box_qpos_adr:self._box_qpos_adr + 3] = new_box_pos
            self.data.qvel[self._box_dof_adr:self._box_dof_adr + 6] = 0.0

    def _get_obs(self):
        # Joint positions in degrees
        joint_pos = np.zeros(6, dtype=np.float32)
        for i, jid in enumerate(self._arm_joint_ids):
            qadr = self.model.jnt_qposadr[jid]
            joint_pos[i] = np.rad2deg(self.data.qpos[qadr])

        # Gripper opening: 0 (closed) to 100 (open)
        fl_qadr = self.model.jnt_qposadr[self._finger_left_jnt]
        gripper_pos = self.data.qpos[fl_qadr]
        gripper_normalized = np.float32(np.clip(gripper_pos / 0.04 * 100.0, 0, 100))

        state = np.append(joint_pos, gripper_normalized).astype(np.float32)

        obs = {"state": state}
        if self.render_images and self._renderer is not None:
            obs["images"] = {
                name: self._renderer.render(self.data, name)
                for name in config.CAMERA_NAMES
            }
        return obs

    def _get_info(self):
        box_pos = self.data.xpos[self._box_body].copy()
        bowl_pos = self.data.xpos[self._bowl_body].copy()
        ee_pos = self.data.site_xpos[self._ee_site].copy()

        # Check if box is in the bowl
        horizontal_dist = np.linalg.norm(box_pos[:2] - bowl_pos[:2])
        box_above_bowl_base = box_pos[2] > bowl_pos[2] - 0.01
        box_below_bowl_rim = box_pos[2] < bowl_pos[2] + config.BOWL_HEIGHT + 0.02
        success = horizontal_dist < config.BOWL_RADIUS and box_above_bowl_base and box_below_bowl_rim

        return {
            "success": bool(success),
            "box_pos": box_pos,
            "bowl_pos": bowl_pos,
            "ee_pos": ee_pos,
            "ee_to_box_dist": float(np.linalg.norm(ee_pos - box_pos)),
            "box_to_bowl_dist": float(np.linalg.norm(box_pos[:2] - bowl_pos[:2])),
        }

    def _compute_reward(self, info):
        reward = 0.0
        # Shaping: encourage approaching the box
        reward -= 0.1 * info["ee_to_box_dist"]
        # Grasp bonus: box is lifted above table
        if info["box_pos"][2] > 0.47:
            reward += 1.0
            # Shaping: encourage moving box toward bowl
            reward -= 0.1 * info["box_to_bowl_dist"]
        # Placement bonus
        if info["success"]:
            reward += 5.0
        return reward

    def _render_human(self):
        """Open an interactive MuJoCo viewer."""
        if self._viewer is None:
            import mujoco.viewer
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self._viewer.sync()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_ee_pos(self):
        """Get end-effector position in world frame."""
        return self.data.site_xpos[self._ee_site].copy()

    def get_box_pos(self):
        """Get box position in world frame."""
        return self.data.xpos[self._box_body].copy()

    def get_bowl_pos(self):
        """Get bowl position in world frame."""
        return self.data.xpos[self._bowl_body].copy()

    @property
    def arm_joint_ids(self):
        return self._arm_joint_ids

    @property
    def ee_site_id(self):
        return self._ee_site

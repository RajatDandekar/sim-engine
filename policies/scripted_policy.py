"""Scripted pick-and-place policy using waypoint-based control with IK.

Uses a shadow copy of MuJoCo data for IK solving, so it doesn't
interfere with the simulation. Outputs normalized actions for the env.
"""

import numpy as np
import mujoco

from .base import BasePolicy
from .. import config


class ScriptedPolicy(BasePolicy):
    """Executes a pre-planned pick-and-place via Cartesian waypoints.

    Approach strategy: retract UP from home first, then move horizontally
    at safe height to avoid sweeping through the workspace and knocking objects.

    Phases:
        0: Retract straight up from home position
        1: Move horizontally to above box at safe height
        2: Lower to pre-grasp
        3: Lower to grasp
        4: Close gripper (wait)
        5: Lift straight up
        6: Move to above bowl at safe height
        7: Lower into bowl
        8: Open gripper (wait)
        9: Retreat up from bowl
       10: Return to home position
       11: Done (hold)
    """

    def __init__(self, env):
        self.env = env
        self.model = env.model
        self.data = env.data

        self._ik_data = mujoco.MjData(self.model)

        self._joint_names = config.JOINT_NAMES[:5]
        self._joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self._joint_names
        ]
        self._dof_ids = [self.model.jnt_dofadr[jid] for jid in self._joint_ids]
        self._ee_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")

        self._phase = 0
        self._phase_step = 0
        self._gripper_target = 1.0
        self._current_joint_targets = None
        self._smoothed_action = None  # for motion smoothing
        self._home_ee = None
        # Per-phase smoothing: higher = faster tracking
        self._phase_smoothing = {
            0: 0.8,   # retract up — fast
            1: 0.8,   # move to above box — fast
            2: 0.7,   # lower to pre-grasp — medium
            3: 0.7,   # lower to grasp — medium
            4: 1.0,   # close gripper — instant
            5: 0.5,   # lift — slower
            6: 0.5,   # transfer to bowl — slower
            7: 0.5,   # lower into bowl — slower
            8: 1.0,   # open gripper — instant
            9: 0.4,   # retreat up — slow
            10: 0.4,  # return to home — slow
        }

        # Heights (world frame z)
        self._safe_z = 0.62
        self._pre_grasp_z = 0.46
        self._grasp_z = 0.434     # EE at box center, finger pads wrap sides
        self._place_z = 0.48

    def reset(self):
        self._phase = 0
        self._phase_step = 0
        self._gripper_target = 1.0
        self._current_joint_targets = None
        self._smoothed_action = None
        self._home_ee = self.env.get_ee_pos()

    def select_action(self, observation: dict) -> np.ndarray:
        box_pos = self.env.get_box_pos()
        bowl_pos = self.env.get_bowl_pos()
        ee_pos = self.env.get_ee_pos()

        if self._phase == 0:
            # Retract straight up from home (keeps x,y fixed)
            target = np.array([self._home_ee[0], self._home_ee[1], self._safe_z])
            self._gripper_target = 1.0
            self._move_toward(target, ee_pos, threshold=0.02)

        elif self._phase == 1:
            # Move horizontally to above box at safe height
            target = np.array([box_pos[0], box_pos[1], self._safe_z])
            self._gripper_target = 1.0
            self._move_toward(target, ee_pos, threshold=0.025)

        elif self._phase == 2:
            # Lower to pre-grasp
            target = np.array([box_pos[0], box_pos[1], self._pre_grasp_z])
            self._gripper_target = 1.0
            self._move_toward(target, ee_pos, threshold=0.025)

        elif self._phase == 3:
            # Lower to grasp
            target = np.array([box_pos[0], box_pos[1], self._grasp_z])
            self._gripper_target = 1.0
            self._move_toward(target, ee_pos, threshold=0.025)

        elif self._phase == 4:
            # Close gripper
            self._gripper_target = -1.0
            self._phase_step += 1
            if self._phase_step > 30:
                self._advance_phase()

        elif self._phase == 5:
            # Lift straight up
            target = np.array([ee_pos[0], ee_pos[1], self._safe_z])
            self._gripper_target = -1.0
            self._move_toward(target, ee_pos, threshold=0.02)

        elif self._phase == 6:
            # Move to above bowl at safe height
            target = np.array([bowl_pos[0], bowl_pos[1], self._safe_z])
            self._gripper_target = -1.0
            self._move_toward(target, ee_pos, threshold=0.02)

        elif self._phase == 7:
            # Lower into bowl
            target = np.array([bowl_pos[0], bowl_pos[1], self._place_z])
            self._gripper_target = -1.0
            self._move_toward(target, ee_pos, threshold=0.02)

        elif self._phase == 8:
            # Open gripper
            self._gripper_target = 1.0
            self._phase_step += 1
            if self._phase_step > 20:
                self._advance_phase()

        elif self._phase == 9:
            # Retreat up from bowl
            target = np.array([bowl_pos[0], bowl_pos[1], self._safe_z])
            self._gripper_target = 1.0
            self._move_toward(target, ee_pos, threshold=0.025)

        elif self._phase == 10:
            # Return to home position
            target = np.array([self._home_ee[0], self._home_ee[1], self._home_ee[2]])
            self._gripper_target = 1.0
            self._move_toward(target, ee_pos, threshold=0.025)

        else:
            # Done — hold home position
            return self._build_action_from_current()

        return self._build_action()

    def _advance_phase(self):
        self._phase += 1
        self._phase_step = 0

    def _move_toward(self, target_pos, ee_pos, threshold=0.02):
        """Solve IK on shadow data, output joint targets."""
        dist = np.linalg.norm(ee_pos - target_pos)

        self._ik_data.qpos[:] = self.data.qpos[:]
        self._ik_data.qvel[:] = 0
        mujoco.mj_forward(self.model, self._ik_data)

        for _ in range(40):
            shadow_ee = self._ik_data.site_xpos[self._ee_site].copy()
            error = target_pos - shadow_ee
            if np.linalg.norm(error) < 0.001:
                break

            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self._ik_data, jacp, None, self._ee_site)
            J = jacp[:, self._dof_ids]

            lam = 0.05
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), error)

            for i, jid in enumerate(self._joint_ids):
                self._ik_data.qpos[self.model.jnt_qposadr[jid]] += dq[i] * 0.4
            mujoco.mj_forward(self.model, self._ik_data)

        self._current_joint_targets = np.array([
            self._ik_data.qpos[self.model.jnt_qposadr[jid]]
            for jid in self._joint_ids
        ])

        self._phase_step += 1
        if dist < threshold or self._phase_step > 200:
            self._advance_phase()

    def _build_action(self) -> np.ndarray:
        raw = np.zeros(7, dtype=np.float32)
        if self._current_joint_targets is not None:
            for i in range(5):
                lo, hi = self.env._ctrl_ranges[i]
                raw[i] = 2.0 * (self._current_joint_targets[i] - lo) / (hi - lo) - 1.0
        raw[5] = self._gripper_target
        raw[6] = self._gripper_target
        raw = np.clip(raw, -1.0, 1.0)

        # Per-phase exponential smoothing for uniform-looking speed
        alpha = self._phase_smoothing.get(self._phase, 0.7)
        if self._smoothed_action is None:
            self._smoothed_action = raw.copy()
        else:
            self._smoothed_action[:5] += alpha * (raw[:5] - self._smoothed_action[:5])
            self._smoothed_action[5:] = raw[5:]  # gripper snaps immediately
        return self._smoothed_action.copy()

    def _build_action_from_current(self) -> np.ndarray:
        action = np.zeros(7, dtype=np.float32)
        for i, jid in enumerate(self._joint_ids):
            lo, hi = self.env._ctrl_ranges[i]
            qpos = self.data.qpos[self.model.jnt_qposadr[jid]]
            action[i] = 2.0 * (qpos - lo) / (hi - lo) - 1.0
        action[5] = self._gripper_target
        action[6] = self._gripper_target
        return np.clip(action, -1.0, 1.0)

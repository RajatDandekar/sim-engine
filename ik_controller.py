"""Simple Jacobian-based IK controller using MuJoCo's built-in functions."""

import mujoco
import numpy as np


class IKController:
    """Resolved-rate IK: computes joint velocities to reach a Cartesian target.

    Uses MuJoCo's analytical Jacobian (mj_jac) + damped least-squares.
    """

    def __init__(self, model, data, site_name="ee_site", joint_names=None, damping=0.05):
        self.model = model
        self.data = data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

        if joint_names is None:
            joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        self.joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names
        ]
        self.dof_ids = [model.jnt_dofadr[jid] for jid in self.joint_ids]
        self.damping = damping

    def compute(self, target_pos, step_size=0.5):
        """Compute joint position deltas to move EE toward target_pos.

        Args:
            target_pos: (3,) desired EE position in world frame
            step_size: scaling factor for the IK step (0-1)

        Returns:
            dq: (n_joints,) joint angle deltas in radians
        """
        mujoco.mj_forward(self.model, self.data)

        # Current EE position
        ee_pos = self.data.site_xpos[self.site_id].copy()
        error = target_pos - ee_pos

        # Compute full Jacobian
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.site_id)

        # Extract columns for our joints only
        J = jacp[:, self.dof_ids]  # (3, n_joints)

        # Damped least-squares pseudoinverse: dq = J^T (J J^T + lambda^2 I)^-1 * error
        JJT = J @ J.T + self.damping ** 2 * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, error)

        return dq * step_size

    def solve_to_position(self, target_pos, max_iters=100, tolerance=0.005):
        """Iteratively solve IK to reach target position.

        Args:
            target_pos: (3,) desired EE position
            max_iters: maximum IK iterations
            tolerance: stop when EE is within this distance (meters)

        Returns:
            target_qpos: (n_joints,) target joint positions in radians
            success: bool
        """
        for _ in range(max_iters):
            mujoco.mj_forward(self.model, self.data)
            ee_pos = self.data.site_xpos[self.site_id]
            dist = np.linalg.norm(target_pos - ee_pos)
            if dist < tolerance:
                break

            dq = self.compute(target_pos, step_size=0.3)
            for i, dof_id in enumerate(self.dof_ids):
                self.data.qpos[self.model.jnt_qposadr[self.joint_ids[i]]] += dq[i]

        mujoco.mj_forward(self.model, self.data)
        final_dist = np.linalg.norm(target_pos - self.data.site_xpos[self.site_id])

        target_qpos = np.array([
            self.data.qpos[self.model.jnt_qposadr[jid]] for jid in self.joint_ids
        ])
        return target_qpos, final_dist < tolerance * 2

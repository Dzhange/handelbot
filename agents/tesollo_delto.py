from copy import deepcopy
from pathlib import Path

import numpy as np
import sapien
import torch
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig, PDEEPoseControllerConfig
from mani_skill.agents.registration import register_agent
from transforms3d.euler import euler2quat

_HERE = Path(__file__).parent
PACKAGE_ASSET_DIR = _HERE.parent / "assets" / "tesollo_delto"

def quat_to_euler_xyz(q):
    # q: (N, 4) in [w, x, y, z] format
    w, x, y, z = q.unbind(dim=-1)

    # roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)

    # yaw (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    return torch.stack((roll, pitch, yaw), dim=-1)

class DeltoBase(BaseAgent):

    disable_self_collisions = False

    joint_stiffness = 1e2
    joint_damping = 1
    joint_force_limit = 5e1
    
    arm_joint_stiffness = 1e3
    arm_joint_damping = 1e2
    arm_joint_force_limit = 100

    hand_dof = 20
    arm_dof = 7

    def get_link_pose(self, link_name):
        return next((l.pose for l in self.robot.links if l.name == link_name), None)

    @property
    def floating_hand_pose(self):
        return self.get_link_pose(self.floating_base_link)

    @property
    def ee_pose(self):
        return self.get_link_pose(self.ee_link_name)

    @property
    def palm_pose(self):
        return self.get_link_pose(self.palm_link)

    @property
    def finger_tip_pose(self):
        return [self.get_link_pose(link_name) for link_name in self.finger_tip_links]

    def get_proprioception(self):
        p = self.ee_pose.p
        ee_q = self.ee_pose.q
        ee_euler = quat_to_euler_xyz(ee_q)
        obs = dict(pose=torch.cat([p, ee_euler, self.robot.get_qpos()[:, 7:]], dim=-1))
        return obs

class DeltoLeftPanda(DeltoBase):
    uid = "delto_left_panda"
    urdf_path = f"{PACKAGE_ASSET_DIR}/urdf/dg5f_left_panda.urdf"
    arm_joint_names = [f'panda_left_joint{i}' for i in range(1, 8)]
    ee_link_name = "panda_left_link8"
    
    urdf_config = dict(
        _materials=dict(
            finger=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
        ),
        link={
            k: dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1 
            )
            for k in [f"lj_dg_{i}_{j}" for i in range(1, 6) for j in [1, 2, 3, 4, 'tip']]
        },
    )
    floating_base_link = "ll_dg_mount"
    palm_link = "ll_dg_palm"
    finger_tip_links = [f"ll_dg_{i}_tip" for i in range(1, 6)]
    hand_joint_names = [f"lj_dg_{i}_{j}" for i in range(1, 6) for j in range(1, 5)]

    @property
    def _controller_configs(self):
        hand_joint_pos = PDJointPosControllerConfig(
            joint_names=self.hand_joint_names,
            lower=None,
            upper=None,
            stiffness=self.joint_stiffness,
            damping=self.joint_damping,
            force_limit=self.joint_force_limit,
            normalize_action=False,
        )

        hand_joint_delta_pos = deepcopy(hand_joint_pos)
        hand_joint_delta_pos.use_delta = True
        hand_joint_delta_pos.normalize_action = True
        hand_joint_delta_pos.lower = -0.1
        hand_joint_delta_pos.upper = 0.1

        arm_pd_ee_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=None,
            pos_upper=None, 
            rot_lower=None,
            rot_upper=None,
            stiffness=self.arm_joint_stiffness,
            damping=self.arm_joint_damping,
            force_limit=self.arm_joint_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            use_delta=False,
            normalize_action=False
        )

        return dict(
            pd_ee_script_xy_pos_joint_delta_pos=dict(
                arm=arm_pd_ee_pose,
                hand=hand_joint_delta_pos,
            ),
        )

@register_agent()
class DeltoLeftPandaReducedFixed1(DeltoLeftPanda):
    uid = "delto_left_panda_reduced_fixed1"
    urdf_path = f"{PACKAGE_ASSET_DIR}/urdf/dg5f_left_panda_reduced_fixed1.urdf"
    hand_joint_names = [f"lj_dg_{i}_{j}" for i in range(1, 6) for j in range(1, 4)]

    keyframes = dict(
        piano_bimanual=Keyframe(
            qpos=np.array(
                [ 
                0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,   
                 ]
            ),
            qvel=np.zeros(22),
            pose=sapien.Pose(p=[0, 0.25, 0], q=euler2quat(0, 0, 0)), 
        ),
    )

class DeltoRightPanda(DeltoLeftPanda):
    uid = "delto_right_panda"
    urdf_path = f"{PACKAGE_ASSET_DIR}/urdf/dg5f_right_panda.urdf"
    arm_joint_names = [f'panda_joint{i}' for i in range(1, 8)]
    ee_link_name = "panda_link8"

    urdf_config = dict(
        _materials=dict(
            finger=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
        ),
        link={
            k: dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1 
            )
            for k in [f"rj_dg_{i}_{j}" for i in range(1, 6) for j in [1, 2, 3, 4, 'tip']]
        },
    )
    floating_base_link = "rl_dg_mount"
    palm_link = "rl_dg_palm"
    finger_tip_links = [f"rl_dg_{i}_tip" for i in range(1, 6)]
    hand_joint_names = [f"rj_dg_{i}_{j}" for i in range(1, 6) for j in range(1, 5)]

@register_agent()
class DeltoRightPandaReducedFixed1(DeltoRightPanda):
    uid = "delto_right_panda_reduced_fixed1"
    urdf_path = f"{PACKAGE_ASSET_DIR}/urdf/dg5f_right_panda_reduced_fixed1.urdf"
    hand_joint_names = [f"rj_dg_{i}_{j}" for i in range(1, 6) for j in range(1, 4)]

    keyframes = dict(
    piano_bimanual=Keyframe(
            qpos=np.array(
                [ 
                0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,
                0, 0, 0,   
                 ]
            ),
            qvel=np.zeros(22),
            pose=sapien.Pose(p=[0, -0.25, 0], q=euler2quat(0, 0, 0)), 
        ),
    )
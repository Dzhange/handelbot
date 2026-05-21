from pathlib import Path

import numpy as np
import sapien
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from transforms3d.euler import euler2quat

_HERE = Path(__file__).parent
PACKAGE_ASSET_DIR = _HERE.parent / "assets" / "shadow_hand"


class ShadowHandBase(BaseAgent):
    """
    Minimal Shadow Hand agent for loading test.

    Goal for now:
    - Register shadow_hand_left / shadow_hand_right as ManiSkill agents
    - Load Shadow Hand URDF
    - Provide the interface that PianoBimanualEnv expects:
      palm_pose, floating_hand_pose, finger_tip_pose
    - Use simple joint position control for the hand joints

    This is not the final training-ready Shadow Hand agent yet.
    """

    disable_self_collisions = False

    joint_stiffness = 1e2
    joint_damping = 1
    joint_force_limit = 5e1

    urdf_path = f"{PACKAGE_ASSET_DIR}/urdf/shadow_hand.urdf"

    # Link names from shadow_hand.urdf
    palm_link = "palm"

    finger_tip_links = [
        "thumb_distal",
        "index_finger_distal",
        "middle_finger_distal",
        "ring_finger_distal",
        "little_finger_distal",
    ]

    # Joint names from shadow_hand.urdf
    # Note: the URDF has a typo: "index_finger_join2" instead of "index_finger_joint2"
    hand_joint_names = [
        "forearm_joint",
        "wrist_joint",

        "thumb_joint1",
        "thumb_joint2",
        "thumb_joint3",
        "thumb_joint4",
        "thumb_joint5",

        "index_finger_joint1",
        "index_finger_join2",
        "index_finger_joint3",
        "index_finger_joint4",

        "middle_finger_joint1",
        "middle_finger_joint2",
        "middle_finger_joint3",
        "middle_finger_joint4",

        "ring_finger_joint1",
        "ring_finger_joint2",
        "ring_finger_joint3",
        "ring_finger_joint4",

        "little_finger_joint1",
        "little_finger_joint2",
        "little_finger_joint3",
        "little_finger_joint4",
        "little_finger_joint5",
    ]

    def get_link_pose(self, link_name):
        return next((link.pose for link in self.robot.links if link.name == link_name), None)

    @property
    def palm_pose(self):
        return self.get_link_pose(self.palm_link)

    @property
    def floating_hand_pose(self):
        # The original Delto agent has floating_hand_pose.
        # For this minimal Shadow Hand test, we use palm_pose as a substitute.
        return self.palm_pose

    @property
    def ee_pose(self):
        # Some parts of the HandelBot environment may expect ee_pose.
        # For a hand-only Shadow Hand agent, use palm_pose for now.
        return self.palm_pose

    @property
    def finger_tip_pose(self):
        return [self.get_link_pose(link_name) for link_name in self.finger_tip_links]

    def get_proprioception(self):
        # Minimal proprioception.
        # The original Delto agent returns pose + qpos.
        # For now, return qpos so the environment can continue loading.
        return dict(qpos=self.robot.get_qpos())

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

        return dict(
            pd_joint_pos=dict(
                hand=hand_joint_pos,
            ),
        )


@register_agent()
class ShadowHandLeft(ShadowHandBase):
    uid = "shadow_hand_left"

    keyframes = dict(
        piano_bimanual=Keyframe(
            qpos=np.zeros(24),
            qvel=np.zeros(24),
            pose=sapien.Pose(
                p=[0, 0.25, 0.2],
                q=euler2quat(0, 0, 0),
            ),
        ),
    )


@register_agent()
class ShadowHandRight(ShadowHandBase):
    uid = "shadow_hand_right"

    keyframes = dict(
        piano_bimanual=Keyframe(
            qpos=np.zeros(24),
            qvel=np.zeros(24),
            pose=sapien.Pose(
                p=[0, -0.25, 0.2],
                q=euler2quat(0, 0, 0),
            ),
        ),
    )

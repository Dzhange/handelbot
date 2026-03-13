import numpy as np
import pyroki as pk
from pyroki.collision import RobotCollision, HalfSpace
from .pyroki_utils import solve_closest_joint_with_collision

class BimanualCollisionChecker:
    def __init__(self, right_urdf, left_urdf, arm_dof=7):
        self.arm_dof = arm_dof
        self.right_urdf = right_urdf
        self.right_robot = pk.Robot.from_urdf(self.right_urdf)
        self.right_coll = RobotCollision.from_urdf(self.right_urdf)

        self.left_urdf = left_urdf
        self.left_robot = pk.Robot.from_urdf(self.left_urdf)
        self.left_coll = RobotCollision.from_urdf(self.left_urdf)
        
        self.plane_coll = HalfSpace.from_point_and_normal(
            np.array([0.095, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])
        )
        _ = solve_closest_joint_with_collision(
            self.left_robot, self.left_coll, np.zeros(16), np.zeros(16),
            [self.plane_coll]
        )
        _ = solve_closest_joint_with_collision(
            self.right_robot, self.right_coll, np.zeros(16), np.zeros(16),
            [self.plane_coll]
        )

    def get_safe_move(self,
                      right_target_hand_qpos,
                      left_target_hand_qpos,
                      right_current_joints=None, left_current_joints=None):
        if right_current_joints is None:
            right_current_joints = right_target_hand_qpos
        if left_current_joints is None:
            left_current_joints = left_target_hand_qpos

        safe_right_qpos = solve_closest_joint_with_collision(
            self.right_robot, self.right_coll, right_target_hand_qpos[4:], right_current_joints[self.arm_dof + 4:],
            [self.plane_coll]
        )

        safe_left_qpos = solve_closest_joint_with_collision(
            self.left_robot, self.left_coll, left_target_hand_qpos[4:], left_current_joints[self.arm_dof + 4:],
            [self.plane_coll]
        )

        safe_right_qpos = np.concatenate([right_target_hand_qpos[:4], safe_right_qpos])
        safe_left_qpos = np.concatenate([left_target_hand_qpos[:4], safe_left_qpos])
        

        return safe_right_qpos, safe_left_qpos
        


from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
import matplotlib.pyplot as plt


@dataclass
class Proprio:
    # supplied as arguments
    eef_pos: np.ndarray
    eef_quat: np.ndarray
    # gripper_open: float  # gripper_width

    # computed in __init__
    # gripper_open_np: np.ndarray  # gripper_width converted to array
    eef_euler: np.ndarray  # rotation in euler
    eef_pos_euler: np.ndarray

    joint_pos: np.ndarray
    joint_vel: np.ndarray

    def __init__(
        self,
        eef_pos: list[float],
        eef_quat: list[float],
        joint_pos: list[float],
        joint_vel: list[float]
        # gripper_open: float,
    ):
        self.eef_pos = np.array(eef_pos)  # , dtype=np.float32)
        self.eef_quat = np.array(eef_quat)  # , dtype=np.float32)
        self.joint_pos = np.array(joint_pos)
        self.joint_vel = np.array(joint_vel)
        # self.gripper_open = gripper_open

        # self.gripper_open_np = np.array([self.gripper_open])  # , dtype=np.float32)
        self.eef_euler = Rotation.from_quat(self.eef_quat).as_euler("xyz")  # .astype(np.float32)
        # TODO: rename this to include gripper
        self.eef_pos_euler = np.concatenate([self.eef_pos, self.eef_euler])


def position_action_to_delta_action(
    curr_pos: np.ndarray, curr_euler: np.ndarray, new_pos: np.ndarray, new_euler: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    delta_pos = new_pos - curr_pos
    curr_rot = Rotation.from_euler("xyz", curr_euler)
    target_rot = Rotation.from_euler("xyz", new_euler)
    delta_rot = target_rot * curr_rot.inv()
    delta_euler = delta_rot.as_euler("xyz")
    return delta_pos, delta_euler


# positional interpolation
def get_waypoint(start_pt, target_pt, max_delta):
    total_delta = target_pt - start_pt
    num_steps = (np.linalg.norm(total_delta) // max_delta) + 1
    remainder = np.linalg.norm(total_delta) % max_delta
    if remainder > 1e-3:
        num_steps += 1
    delta = total_delta / num_steps

    def gen_waypoint(i):
        return start_pt + delta * min(i, num_steps)

    return gen_waypoint, int(num_steps)


# rotation interpolation
def get_ori(initial_euler, final_euler, num_steps):
    diff = np.linalg.norm(final_euler - initial_euler)
    ori_chg = Rotation.from_euler("xyz", [initial_euler.copy(), final_euler.copy()], degrees=False)
    if diff < 0.02 or num_steps < 2:

        def gen_ori(i):
            return initial_euler

    else:
        slerp = Slerp([1, num_steps], ori_chg)

        def gen_ori(i):
            interp_euler = slerp(i).as_euler("xyz")
            return interp_euler

    return gen_ori

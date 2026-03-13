
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple
import re
import math
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class SingleJointInfo:
	time: float
	qpos: float
	effort: float = None

@dataclass
class SinglePoseInfo:
	time: float
	pose: List[float] # 7 dim

class FingerTracker:

	def __init__(self, joint_names):
		self.joint_names = joint_names

		self._actual: Dict[str, List[SingleJointInfo]] = {n: [] for n in joint_names}
		self._target: Dict[str, List[SingleJointInfo]] = {n: [] for n in joint_names}

	def add_target(self, qpos: List[float], time):

		for name, q in zip(self.joint_names, qpos):
			self._target[name].append(
				SingleJointInfo(time=float(time), qpos=float(q), effort=None)
			)

	def add_actual(self, qpos: List[float], effort: List[float], time):

		for name, q, f in zip(self.joint_names, qpos, effort):
			self._actual[name].append(
				SingleJointInfo(time=float(time), qpos=float(q), effort=float(f))
			)

	def name_to_fj(self, name):
		return (int(name[6]), int(name[8]))

	def plot(self, output_path) -> None:
		rows, cols = 5, 4  # 5 fingers, 4 joints each
		fig, axes = plt.subplots(rows, cols, figsize=(16, 20), sharex=True)
		axes = axes.reshape(rows, cols)

		for joint_name in self.joint_names:
			finger_idx, joint_idx = self.name_to_fj(joint_name)
			ax = axes[finger_idx - 1, joint_idx - 1]

			actual_data = self._actual[joint_name]
			if actual_data:
				actual_times = [info.time for info in actual_data]
				actual_qpos = [info.qpos for info in actual_data]
				ax.plot(actual_times, actual_qpos, 'b-', label='Actual qpos', linewidth=1.5)

				actual_effort = [info.effort for info in actual_data]
				ax2 = ax.twinx()
				ax2.plot(actual_times, actual_effort, 'g--', label='Effort', linewidth=1.5)
				if joint_idx == cols:  
					ax2.set_ylabel('Effort', fontsize=9)

			target_data = self._target[joint_name]
			if target_data:
				target_times = [info.time for info in target_data]
				target_qpos = [info.qpos for info in target_data]
				ax.plot(target_times, target_qpos, 'r-', label='Target qpos', linewidth=1.5)


			ax.set_title(f'{finger_idx}_{joint_idx}', fontsize=10)
			ax.grid(True, alpha=0.3)

			if joint_idx == 1:
				ax.set_ylabel('Joint Position (rad)', fontsize=9)
			if finger_idx == rows:
				ax.set_xlabel('Time (s)', fontsize=9)

			handles1, labels1 = ax.get_legend_handles_labels()
			if 'ax2' in locals():
			    handles2, labels2 = ax2.get_legend_handles_labels()
			    ax.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc='upper left')
			else:
			    ax.legend(fontsize=8, loc='upper left')

		plt.tight_layout()
		plt.savefig(output_path, bbox_inches='tight')


		rows, cols = 5, 4  # 5 fingers, 4 joints each
		fig, axes = plt.subplots(rows, cols, figsize=(16, 20), sharex=True)
		axes = axes.reshape(rows, cols)

		for joint_name in self.joint_names:
			finger_idx, joint_idx = self.name_to_fj(joint_name)
			ax = axes[finger_idx - 1, joint_idx - 1]

			actual_data = self._actual[joint_name]
			target_data = self._target[joint_name]

			time = [info.time for info in actual_data]
			target_qpos = np.array([info.qpos for info in target_data])
			actual_qpos = np.array([info.qpos for info in actual_data])

			# desired 
			
			desired = target_qpos[1:] - actual_qpos[:-1]
			ax.plot(time[:-1], desired, 'b-', label='desired', linewidth=1.5)

			error = target_qpos - actual_qpos

			ax.plot(time, error, 'r-', label='error', linewidth=1.5)


			ax.set_title(f'{finger_idx}_{joint_idx}', fontsize=10)
			ax.grid(True, alpha=0.3)

			if joint_idx == 1:
				ax.set_ylabel('Joint Position (rad)', fontsize=9)
			if finger_idx == rows:
				ax.set_xlabel('Time (s)', fontsize=9)

			handles1, labels1 = ax.get_legend_handles_labels()
			if 'ax2' in locals():
			    handles2, labels2 = ax2.get_legend_handles_labels()
			    ax.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc='upper left')
			else:
			    ax.legend(fontsize=8, loc='upper left')

		plt.tight_layout()
		path = output_path.replace('.jpg', '_diff.jpg')
		plt.savefig(path, bbox_inches='tight')






class ArmTracker:

	def __init__(self, joint_names):
		self.joint_names = joint_names

		self._actual: Dict[str, List[SingleJointInfo]] = {n: [] for n in joint_names}
		self._target_ee: List[SinglePoseInfo] = []
		self._actual_ee: List[SinglePoseInfo] = [] # To store actual FK EE if available

	def add_target_ee(self, pose: List[float], time):
		self._target_ee.append(
			SinglePoseInfo(time=float(time), pose=list(pose))
		)

	# NOTE: We assume we might get actual EE pose from outside, or we just don't plot error if not available?
	# But request asks to plot ERROR between target ee and actual.
	# We don't have actual EE here, only joint angles.
	# We need actual EE pose.

	def add_actual_ee(self, pose: List[float], time):
		self._actual_ee.append(
			SinglePoseInfo(time=float(time), pose=list(pose))
		)

	def add_actual(self, qpos: List[float], time):
		for name, q in zip(self.joint_names, qpos):
			self._actual[name].append(
				SingleJointInfo(time=float(time), qpos=float(q))
			)

	def plot(self, output_path) -> None:
		# Plot joint angles (actual only)
		rows, cols = 3, 3 
		fig, axes = plt.subplots(rows, cols, figsize=(12, 12), sharex=True)
		axes = axes.reshape(rows, cols)

		for joint_name in self.joint_names:
			joint_idx = int(joint_name[-1])
			row_idx = joint_idx // 3 
			col_idx = joint_idx % 3
			ax = axes[row_idx, col_idx]

			actual_data = self._actual[joint_name]
			if actual_data:
				actual_times = [info.time for info in actual_data]
				actual_qpos = [info.qpos for info in actual_data]
				ax.plot(actual_times, actual_qpos, 'b-', label='Actual qpos', linewidth=1.5)

			ax.set_title(joint_name, fontsize=10)
			ax.grid(True, alpha=0.3)

			if col_idx == 0:
				ax.set_ylabel('Joint Position (rad)', fontsize=9)
			if row_idx == rows-1:
				ax.set_xlabel('Time (s)', fontsize=9)

			ax.legend(fontsize=8, loc='upper left')

		plt.tight_layout()
		plt.savefig(output_path, bbox_inches='tight')

		# Plot EE Error if both target and actual EE are available
		if self._target_ee: # and self._actual_ee:
			# Assuming time steps align or are close enough?
			# Or just plot trajectories.
			# Request says "plot the error".
			# If we have actual EE, we calculate error.
			# If we don't (yet), we just plot target?

			# We need actual EE to plot error.
			# I will add support for storing actual EE.

			fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
			axes2 = axes2.flatten()
			target_times = [info.time for info in self._target_ee]
			target_poses = np.array([info.pose for info in self._target_ee]) # N x 7

			labels = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']

			if self._actual_ee:
				actual_times = [info.time for info in self._actual_ee]
				actual_poses = np.array([info.pose for info in self._actual_ee])

				# Error plot? Or trajectory comparison?
				# Trajectory comparison first.
				for i in range(6):
					ax = axes2[i]
					ax.plot(target_times, target_poses[:, i], 'r--', label=f'Target {labels[i]}')
					# We assume len matches or we just plot both vs time
					ax.plot(actual_times, actual_poses[:, i], 'b-', label=f'Actual {labels[i]}')
					ax.set_title(f'EE {labels[i]}')
					ax.grid(True)
					ax.legend()

				# Plot error in the 8th subplot (position error)
				ax_err = axes2[7]
				# Only if lengths match exactly, otherwise interpolation needed.
				# For simplicity, assume strict step matching if times match?
				# Or just skip error plot if lengths mismatch significantly.
				if len(target_poses) == len(actual_poses):
					pos_error = np.linalg.norm(target_poses[:, :3] - actual_poses[:, :3], axis=1)
					ax_err.plot(target_times, pos_error, 'k-', label='Pos Error')
					ax_err.set_title('Position Error (m)')
					ax_err.grid(True)
			else:
				# Just target
				for i in range(7):
					ax = axes2[i]
					ax.plot(target_times, target_poses[:, i], 'r-', label=f'Target {labels[i]}')
					ax.set_title(f'EE {labels[i]}')
					ax.grid(True)

			plt.tight_layout()
			path_ee = output_path.replace('.jpg', '_ee.jpg')
			plt.savefig(path_ee, bbox_inches='tight')
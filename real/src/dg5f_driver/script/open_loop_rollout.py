import os
import time
import csv
import yaml
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pyrallis
import rclpy
import yourdfpy
from rclpy.node import Node

from real.src.dg5f_driver.utils.bidex_env import BiDexEnv, BiDexEnvConfig, RealPianoSeparateWrapper
from real.src.dg5f_driver.utils.safe_utils import BimanualCollisionChecker

from real.src.dg5f_driver.utils.constants import RIGHT_JOINT_LIMITS, LEFT_JOINT_LIMITS

@dataclass(frozen=True)
class DataConfig:
    root: str
    run_dir: str
    left_folder: str
    right_folder: str

    @property
    def right_urdf_path(self):
        return os.path.join(self.root, "assets/tesollo_delto/urdf/dg5f_right_nothumb.urdf")

    @property
    def left_urdf_path(self):
        return os.path.join(self.root, "assets/tesollo_delto/urdf/dg5f_left_nothumb.urdf")

    @property
    def open_loop_right_hand_path(self):
        return f"{self.root}/real/ckpts/{self.right_folder}/right_hand_target_qpos.npy"

    @property
    def open_loop_left_hand_path(self):
        return f"{self.root}/real/ckpts/{self.left_folder}/left_hand_target_qpos.npy"

    @property
    def open_loop_right_arm_path(self):
        return f"{self.root}/real/ckpts/{self.right_folder}/right_arm_target_ee_pose.npy"

    @property
    def open_loop_left_arm_path(self):
        return f"{self.root}/real/ckpts/{self.left_folder}/left_arm_target_ee_pose.npy"

    @property
    def bidex_cfg_path(self):
        return os.path.join(self.root, "real/src/dg5f_driver/config/rl_bidex_env_config.yaml")


class OpenLoopRollout(Node):
    def __init__(self):
        super().__init__('open_loop_rollout')

        # Load configs
        config_path = os.path.join(os.path.dirname(__file__), "../config/open_loop_rollout.yaml")
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        self.args = SimpleNamespace(**config_data)
        run_dir = f"{self.args.rollout_root}/{self.args.folder}/"
        os.makedirs(run_dir, exist_ok=True)
        self.cfg = DataConfig(
            root=self.args.root,
            run_dir=run_dir,
            left_folder=self.args.left_folder,
            right_folder=self.args.right_folder,
        )
        self.track_out_dir = os.path.join(run_dir, "tracking")
        self.results_dir = os.path.join(self.args.root, "rollouts_rl", "tuning", f"real_results_{int(time.time())}")
        self.results_csv_path = os.path.join(self.results_dir, "stats.csv")
        self.get_logger().info(f"Root path: {self.cfg.root}")
        self.get_logger().info(f"Left Offsets to test: {self.args.left_offsets}")
        self.get_logger().info(f"Right Offsets to test: {self.args.right_offsets}")
        if not os.path.exists(self.track_out_dir):
            os.makedirs(self.track_out_dir, exist_ok=True)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)
        if not os.path.exists(self.results_csv_path):
             with open(self.results_csv_path, 'w', newline='') as f:
                 writer = csv.writer(f)
                 writer.writerow(['lh_f1', 'rh_f1', 'l_offset_x', 'l_offset_y', 'l_offset_z', 'r_offset_x', 'r_offset_y', 'r_offset_z', 'f1', 'precision', 'recall', 'lh_precision', 'lh_recall', 'rh_precision', 'rh_recall', 'left_run_folder', 'right_run_folder'])

        # Load Environment
        input("Ready to start? Press any key to continue, or Ctrl+C to stop.")
        config_path_rl = os.path.join(os.path.dirname(__file__), "../config/residual_rl_separate.yaml")
        with open(config_path_rl, "r") as f:
            config_data_rl = yaml.safe_load(f)
        for k, v in config_data_rl.items():
            if not hasattr(self.args, k):
                setattr(self.args, k, v)
        bidex_cfg = pyrallis.load(BiDexEnvConfig, open(self.cfg.bidex_cfg_path, "r"))
        self.env = RealPianoSeparateWrapper(
            BiDexEnv(bidex_cfg, self), 
            sheet_music_path=self.args.sheet_music, 
            device=self.args.device,
        )
        self.setup_collision_checker()
        self.setup_trajectories(bidex_cfg)

        input("Moving the robot to the start position! Press any key to continue, or Ctrl+C to stop.")
        self.apply_offset_to_targets(self.args.left_offsets[0], self.args.right_offsets[0])
        self.env.data_len = self.data_len
        self.get_logger().info(f"Loaded open loop data with {self.data_len} steps")
        self.init(first=True)

        # Timer
        input("beginning open loop rollout! Press any key to continue, or Ctrl+C to stop.")
        self.policy_timer = self.create_timer(1 / bidex_cfg.policy_freq, self.policy_timer_callback)
        self.start_time = self.get_clock().now().nanoseconds
        self.warmup_done = False
        self.warmup_start_time = None
        self.resetting = False


    def apply_offset_to_targets(self, left_offset, right_offset):
        l_x, l_y, l_z = left_offset
        r_x, r_y, r_z = right_offset
        self.get_logger().info(f"Applying Left Offset: {left_offset}, Right Offset: {right_offset}")

        self.target_qpos_hand_right = self._raw_target_qpos_hand_right.copy()
        self.target_ee_pose_arm_right = self._raw_target_ee_pose_arm_right.copy()
        
        self.target_qpos_hand_left = self._raw_target_qpos_hand_left.copy()
        self.target_ee_pose_arm_left = self._raw_target_ee_pose_arm_left.copy()

        # Calculate absolute positions relative to config base + current iteration offset
        base_x_r, base_y_r, base_z_r = self.args.starting_pos_r
        base_x_l, base_y_l, base_z_l = self.args.starting_pos_l

        # Right Hand
        REAL_X_R = base_x_r + r_x
        REAL_Y_R = base_y_r + r_y
        REAL_Z_R = base_z_r + r_z
        
        # Left Hand
        REAL_X_L = base_x_l + l_x
        REAL_Y_L = base_y_l + l_y
        REAL_Z_L = base_z_l + l_z

        # Normalize to start from 0 relative to first frame, then add REAL X/Y offset
        self.target_ee_pose_arm_right[:, 0] -= self.target_ee_pose_arm_right[0, 0]
        self.target_ee_pose_arm_left[:, 0] -= self.target_ee_pose_arm_left[0, 0]
        
        self.target_ee_pose_arm_right[:, 0] += REAL_X_R
        self.target_ee_pose_arm_left[:, 0] += REAL_X_L

        self.target_ee_pose_arm_right[:, 2] = REAL_Z_R - 0.01
        self.target_ee_pose_arm_left[:, 2] = REAL_Z_L

        # Y adjustment
        self.target_ee_pose_arm_right[:, 1] += REAL_Y_R
        self.target_ee_pose_arm_left[:, 1] += REAL_Y_L
        
        self.data_len = min(len(self.target_qpos_hand_right), len(self.target_qpos_hand_left))
        if hasattr(self, 'env'):
             self.env.data_len = self.data_len

    def _now_s(self):
        return (self.get_clock().now().nanoseconds - self.start_time) / 1e9

    def init(self, first=False):
        obs = self.env.reset()
        obs_left, obs_right = obs

        # Initial States
        init_hand_r = self.target_qpos_hand_right[0]
        init_ee_r = self.target_ee_pose_arm_right[0] # 7D: pos(3) + quat(4)
        init_hand_l = self.target_qpos_hand_left[0]
        init_ee_l = self.target_ee_pose_arm_left[0]
        ee_pos_r = init_ee_r[:3]
        ee_pos_l = init_ee_l[:3]

        # Hand Control
        self.env.publish_right_hand(init_hand_r)
        self.env.publish_left_hand(init_hand_l)

        # Arm Control
        if first:
            proprio_r = self.env.get_right_arm_proprio()
            ee_euler_r = proprio_r.eef_euler
            self.default_ee_euler_r = ee_euler_r
        else: 
            ee_euler_r = self.default_ee_euler_r
        self.env.right_move_to(ee_pos_r, ee_euler_r, 10)
        if first:
            proprio_l = self.env.get_left_arm_proprio()
            ee_euler_l = proprio_l.eef_euler
            self.default_ee_euler_l = ee_euler_l
        else:
            ee_euler_l = self.default_ee_euler_l
        self.env.left_move_to(ee_pos_l, ee_euler_l, 10)

    def warmup(self):
        current_time = self.get_clock().now().nanoseconds
        if self.warmup_start_time is None:
            self.warmup_start_time = current_time
            self.get_logger().info("Starting warmup...")

        elapsed = (current_time - self.warmup_start_time) / 1e9
        if elapsed < self.args.warmup_duration_s:
            return 

        self.warmup_done = True
        self.get_logger().info("Warmup complete. Starting (open) control loop...")


    def policy_timer_callback(self):
        if not self.warmup_done:
            self.warmup()
        elif self.resetting:
            self.resetting = False
            self.warmup_done = False

            if hasattr(self.env, 'episode_metrics'):
                m = self.env.episode_metrics
                curr_l_offset = self.args.left_offsets[self.offset_idx]
                curr_r_offset = self.args.right_offsets[self.offset_idx]
                
                print(f"Finished offset {self.offset_idx}: L={curr_l_offset}, R={curr_r_offset}")
                print(f"Metrics: {m}")
                
                 # Write to CSV
                with open(self.results_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        f"{m.get('lh_f1', 0):.4f}",
                        f"{m.get('rh_f1', 0):.4f}",
                        f"{curr_l_offset[0]:.3f}",
                        f"{curr_l_offset[1]:.3f}",
                        f"{curr_l_offset[2]:.3f}",
                        f"{curr_r_offset[0]:.3f}",
                        f"{curr_r_offset[1]:.3f}",
                        f"{curr_r_offset[2]:.3f}",
                        f"{m.get('f1', 0):.4f}",
                        f"{m.get('precision', 0):.4f}",
                        f"{m.get('recall', 0):.4f}",
                        f"{m.get('lh_precision', 0):.4f}",
                        f"{m.get('lh_recall', 0):.4f}",
                        f"{m.get('rh_precision', 0):.4f}",
                        f"{m.get('rh_recall', 0):.4f}",
                        self.cfg.left_folder,
                        self.cfg.right_folder
                    ])
                
                # Plot results
                plot_name = f"plot_offset_{self.offset_idx}_L_{curr_l_offset}_R_{curr_r_offset}.jpg"
                plot_path = os.path.join(self.results_dir, plot_name)
                self.env.plot_notes(plot_path)

                self.offset_idx += 1
                if self.offset_idx >= len(self.args.left_offsets):
                    self.get_logger().info("All offsets tested. Shutting down.")
                    self.shutdown()
                    return
                else:
                    new_l_offset = self.args.left_offsets[self.offset_idx]
                    new_r_offset = self.args.right_offsets[self.offset_idx]
                    self.get_logger().info(f"Moving to next offset ({self.offset_idx + 1}/{len(self.args.left_offsets)}): L={new_l_offset}, R={new_r_offset}")
                    self.apply_offset_to_targets(new_l_offset, new_r_offset)
                    self.env.env.go_home() 
                    self.init() 
        else:
            self.policy_loop()

    def policy_loop(self):
        # Get targets for current step (Joint Space for hand, EE Space for arm)
        target_hand_r = self.target_qpos_hand_right[self.env.song_t_idx].copy()
        target_hand_l = self.target_qpos_hand_left[self.env.song_t_idx].copy()
        ee_pos_r = self.target_ee_pose_arm_right[self.env.song_t_idx][:3]
        ee_euler_r = self.default_ee_euler_r
        ee_pos_l = self.target_ee_pose_arm_left[self.env.song_t_idx][:3]
        ee_euler_l = self.default_ee_euler_l

        # Get current states for safety check
        curr_r_arm_proprio = self.env.get_right_arm_proprio()
        curr_r_hand_proprio = self.env.get_right_hand_proprio()
        curr_l_arm_proprio = self.env.get_left_arm_proprio()
        curr_l_hand_proprio = self.env.get_left_hand_proprio()

        curr_r = None
        if curr_r_arm_proprio and curr_r_hand_proprio is not None:
             curr_r = np.concatenate([curr_r_arm_proprio.joint_pos, curr_r_hand_proprio])
        
        curr_l = None
        if curr_l_arm_proprio and curr_l_hand_proprio is not None:
             curr_l = np.concatenate([curr_l_arm_proprio.joint_pos, curr_l_hand_proprio])
        
        target_hand_r = np.clip(target_hand_r, -0.3, RIGHT_JOINT_LIMITS[:, 1])
        target_hand_l = np.clip(target_hand_l, -0.3, LEFT_JOINT_LIMITS[:, 1])

        # Check safety of Actions
        target_hand_r, target_hand_l = self.collision_checker.get_safe_move(
            target_hand_r, target_hand_l,
            curr_r, curr_l
        )
        target_hand_r = np.clip(target_hand_r, -0.3, RIGHT_JOINT_LIMITS[:, 1])
        target_hand_l = np.clip(target_hand_l, -0.3, LEFT_JOINT_LIMITS[:, 1])
        
        # Move robot
        (next_obs_left, next_obs_right), (rew_left, rew_right), done, info = self.env.step(
            right_arm=(ee_pos_r, ee_euler_r),
            left_arm=(ee_pos_l, ee_euler_l),
            right_hand=target_hand_r,
            left_hand=target_hand_l,
        )
        if done:
            self.resetting = True
        
        self.global_idx += 1
        

    def setup_collision_checker(self):
        right_urdf = yourdfpy.URDF.load(self.cfg.right_urdf_path)
        left_urdf = yourdfpy.URDF.load(self.cfg.left_urdf_path) 
        self.collision_checker = BimanualCollisionChecker(
            right_urdf=right_urdf,
            left_urdf=left_urdf,
        )

    def setup_trajectories(self, bidex_cfg):
        self.global_idx = 0
        self.global_ep = 0
        self.data_len = 0
        self.offset_idx = 0

        control_scale = 10 // bidex_cfg.policy_freq
        self._raw_target_qpos_hand_right = np.load(self.cfg.open_loop_right_hand_path)[:-1][::control_scale]
        self._raw_target_ee_pose_arm_right = np.load(self.cfg.open_loop_right_arm_path)[:-1][::control_scale]

        self._raw_target_qpos_hand_left = np.load(self.cfg.open_loop_left_hand_path)[:-1][::control_scale]
        self._raw_target_ee_pose_arm_left = np.load(self.cfg.open_loop_left_arm_path)[:-1][::control_scale]

        # Standardize hand state from 15 to 20 DoF by adding fixed 4th joint value (1.0)
        if self._raw_target_qpos_hand_right.shape[1] == 15:
            N = self._raw_target_qpos_hand_right.shape[0]
            hand_reshaped = self._raw_target_qpos_hand_right.reshape(N, 5, 3)
            ones = np.ones((N, 5, 1), dtype=self._raw_target_qpos_hand_right.dtype)
            hand_concat = np.concatenate([hand_reshaped, ones], axis=2)
            self._raw_target_qpos_hand_right = hand_concat.reshape(N, 20)

        if self._raw_target_qpos_hand_left.shape[1] == 15:
            N = self._raw_target_qpos_hand_left.shape[0]
            hand_reshaped = self._raw_target_qpos_hand_left.reshape(N, 5, 3)
            ones = np.ones((N, 5, 1), dtype=self._raw_target_qpos_hand_left.dtype)
            hand_concat = np.concatenate([hand_reshaped, ones], axis=2)
            self._raw_target_qpos_hand_left = hand_concat.reshape(N, 20)

        self._raw_target_qpos_hand_right = self._raw_target_qpos_hand_right
        self._raw_target_ee_pose_arm_right = self._raw_target_ee_pose_arm_right
        self._raw_target_qpos_hand_left = self._raw_target_qpos_hand_left
        self._raw_target_ee_pose_arm_left = self._raw_target_ee_pose_arm_left

    def shutdown(self):
        self.get_logger().info("Shutting down...")
        if self.env:
            self.env.plot(self.cfg.track_out_dir)
            if hasattr(self.env, 'shutdown'):
                 self.env.shutdown()
        print(f"wrote to {self.results_csv_path}")
        
        self.stop_keyboard_thread = True
        exit()


def main(args=None):
    rclpy.init(args=args)
    node = OpenLoopRollout()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

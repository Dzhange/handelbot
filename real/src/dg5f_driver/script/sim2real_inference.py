import csv
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pyrallis
import rclpy
import torch
import yaml
import yourdfpy
from rclpy.node import Node
from control_msgs.msg import MultiDOFCommand

from real.src.dg5f_driver.utils.bidex_env import BiDexEnv, BiDexEnvConfig, Sim2RealWrapper
from real.src.dg5f_driver.utils.safe_utils import BimanualCollisionChecker
from real.src.dg5f_driver.utils.constants import RIGHT_JOINT_LIMITS, LEFT_JOINT_LIMITS
from rl.piano_ppo_fast import Agent

@dataclass(frozen=True)
class InferenceConfig:
    root: str
    folder: str
    ckpt_num: int
    
    @property
    def exp_dir(self):
        return os.path.join(self.root, "real/ckpts", self.folder)

    @property
    def checkpoint_path(self):
        return os.path.join(self.exp_dir, f"ckpt_{self.ckpt_num}.pt")

    @property
    def left_arm_traj_path(self):
        path = os.path.join(self.exp_dir, "left_arm_target_ee_pose.npy")
        if not os.path.exists(path):
            path = os.path.join(self.exp_dir, "lef_arm_target_ee_pose.npy")
        return path

    @property
    def right_arm_traj_path(self):
        return os.path.join(self.exp_dir, "right_arm_target_ee_pose.npy")

    @property
    def left_hand_traj_path(self):
        return os.path.join(self.exp_dir, "left_hand_target_qpos.npy")

    @property
    def right_hand_traj_path(self):
        return os.path.join(self.exp_dir, "right_hand_target_qpos.npy")

    @property
    def bidex_cfg_path(self):
        return os.path.join(self.root, "real/src/dg5f_driver/config/rl_bidex_env_config.yaml")

    @property
    def right_urdf_path(self):
        return os.path.join(self.root, "assets/tesollo_delto/urdf/dg5f_right_nothumb.urdf")

    @property
    def left_urdf_path(self):
        return os.path.join(self.root, "assets/tesollo_delto/urdf/dg5f_left_nothumb.urdf")



class Sim2RealInference(Node):
    def __init__(self):
        super().__init__('sim2real_inference')
        
        # Load config from YAML
        config_path = os.path.join(os.path.dirname(__file__), "../config/sim2real_inference.yaml")
        with open(config_path, "r") as f:
            self.config_dict = yaml.safe_load(f)
        self.args = SimpleNamespace(**self.config_dict)
        
        self.cfg = InferenceConfig(
            root=self.args.root,
            folder=self.args.folder,
            ckpt_num=self.args.ckpt_num
        )
        
        self.get_logger().info(f"Root path: {self.cfg.root}")
        self.get_logger().info(f"Loading checkpoint: {self.cfg.checkpoint_path}")
        self.setup_csv_logger()

        # 1. Load Environment
        with open(self.cfg.bidex_cfg_path, "r") as f:
            bidex_dict = yaml.safe_load(f)
        bidex_cfg = pyrallis.decode(BiDexEnvConfig, bidex_dict)
        bidex_cfg.policy_freq = self.args.policy_freq
        self.wrapper = Sim2RealWrapper(
            BiDexEnv(bidex_cfg, self),
            sheet_music_path=self.args.sheet_music,
            device=self.args.device,
            use_sim_obs=self.args.use_sim_obs
        )
        self.wrapper.results_dir = self.results_dir
        self.get_logger().info("Environment and Wrapper initialized")

        # 2. Load Agent
        ckpt = torch.load(self.cfg.checkpoint_path, map_location=self.args.device)
        n_obs, n_act = self.args.obs_dim, self.args.action_dim
        self.agent = Agent(n_obs, n_act, device=self.args.device)
        self.agent.load_state_dict(ckpt)
        self.agent.eval()
        self.get_logger().info("Agent loaded and set to eval mode")

        self.setup_trajectories()
        self.wrapper.data_len = self.data_len

        input("Moving the robot to the start position! Press any key to continue, or Ctrl+C to stop.")
        self.apply_offset_to_targets(self.args.left_offsets[0], self.args.right_offsets[0])
        self.setup_collision_checker()

        # 4. Initialization
        self.traj_idx = 0
        self.num_trajectories = len(self.args.left_offsets)
        self.resetting = False
        self.init_robot()
        
        # 5. Control Loop
        self.timer = self.create_timer(1.0 / self.args.policy_freq, self.control_callback)
        self.obs = self.wrapper.reset()
        self.get_logger().info("Control loop started")

        self.all_actions = []
        self.all_obs = []


    def setup_trajectories(self):
        # --- Arm Trajectories (Scripted EE) ---
        song_scale = max(1, 10 // self.args.policy_freq)
        self._raw_target_ee_pose_arm_left = np.load(self.cfg.left_arm_traj_path)[:-1][::song_scale]
        self._raw_target_ee_pose_arm_right = np.load(self.cfg.right_arm_traj_path)[:-1][::song_scale]
        self.data_len = min(len(self._raw_target_ee_pose_arm_left), len(self._raw_target_ee_pose_arm_right))

    def setup_collision_checker(self):
        right_urdf = yourdfpy.URDF.load(self.cfg.right_urdf_path)
        left_urdf = yourdfpy.URDF.load(self.cfg.left_urdf_path)
        
        self.collision_checker = BimanualCollisionChecker(
            right_urdf=right_urdf,
            left_urdf=left_urdf,
        )

    def setup_csv_logger(self):
        self.results_dir = os.path.join(self.cfg.root, "rollouts_rl", "tuning", f"real_inference_{int(time.time())}")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir, exist_ok=True)
        with open(os.path.join(self.results_dir, "config.yaml"), "w") as f:
            yaml.dump(self.config_dict, f)
            
        self.results_csv_path = os.path.join(self.results_dir, "stats.csv")
        with open(self.results_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['lh_f1', 'rh_f1', 'f1', 'precision', 'recall', 'lh_precision', 'lh_recall', 'rh_precision', 'rh_recall', 'folder', 'ckpt_num'])

    def apply_offset_to_targets(self, left_offset, right_offset):
        x_off_left, y_off_left, z_off_left = left_offset
        x_off_right, y_off_right, z_off_right = right_offset

        self.get_logger().info(f"Applying Offsets - Left: [{x_off_left}, {y_off_left}, {z_off_left}], Right: [{x_off_right}, {y_off_right}, {z_off_right}]")

        self.target_ee_pose_arm_left = self._raw_target_ee_pose_arm_left.copy()
        self.target_ee_pose_arm_right = self._raw_target_ee_pose_arm_right.copy()

        REAL_X_R = 0.525 + x_off_right 
        REAL_Z_R = 0.27 + z_off_right 

        REAL_X_L = 0.525 + x_off_left
        REAL_Z_L = 0.27 + z_off_left 

        self.target_ee_pose_arm_right[:, 0] -= self.target_ee_pose_arm_right[0, 0]
        self.target_ee_pose_arm_left[:, 0] -= self.target_ee_pose_arm_left[0, 0]

        self.target_ee_pose_arm_right[:, 0] += REAL_X_R
        self.target_ee_pose_arm_left[:, 0] += REAL_X_L

        self.target_ee_pose_arm_right[:, 2] = REAL_Z_R - 0.01 
        self.target_ee_pose_arm_left[:, 2] = REAL_Z_L 

        self.target_ee_pose_arm_right[:, 1] += (0.1 + y_off_right) 
        self.target_ee_pose_arm_left[:, 1] += (-0.1 + y_off_left) 

        # Update wrapper offsets for observation transformation
        self.wrapper.obs_offsets = {
            'left_x': REAL_X_L,
            'left_y': -0.1 + y_off_left,
            'left_z': REAL_Z_L,
            'right_x': REAL_X_R,
            'right_y': 0.1 + y_off_right,
            'right_z': REAL_Z_R - 0.01
        }

        self.wrapper.data_len = min(len(self.target_ee_pose_arm_left), len(self.target_ee_pose_arm_right))

    def init_robot(self, first=True):
        self.get_logger().info("Initializing robot to starting position...")
        self.wrapper.reset()
        
        # Initial targets
        idx = 0
        target_right_arm_pos = self.target_ee_pose_arm_right[idx][:3]
        target_left_arm_pos = self.target_ee_pose_arm_left[idx][:3]

        # Capture default euler angles for scripted arm control
        if first:
            proprio_r = self.wrapper.get_right_arm_proprio()
            proprio_l = self.wrapper.get_left_arm_proprio()
            self.default_ee_euler_r = proprio_r.eef_euler
            self.default_ee_euler_l = proprio_l.eef_euler
        
        start = np.zeros(20)
        start[3::4] = 1
        self.wrapper.publish_right_hand(start)
        self.wrapper.publish_left_hand(start)

        # Move to starting position
        self.wrapper.right_move_to(target_right_arm_pos, self.default_ee_euler_r, control_freq=10)
        self.wrapper.left_move_to(target_left_arm_pos, self.default_ee_euler_l, control_freq=10)

        if first:
            input("Ready to start? Press Enter to continue...")

    def control_callback(self):
        if self.traj_idx >= self.num_trajectories:
            self.shutdown()
            exit()
        if self.resetting:
            self.resetting = False
            
            m = self.wrapper.episode_metrics
            self.get_logger().info(f"Finished trajectory {self.traj_idx + 1}/{self.num_trajectories}")
            self.get_logger().info(f"Metrics: {m}")
            with open(self.results_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    f"{m.get('lh_f1', 0):.4f}",
                    f"{m.get('rh_f1', 0):.4f}",
                    f"{m.get('f1', 0):.4f}",
                    f"{m.get('precision', 0):.4f}",
                    f"{m.get('recall', 0):.4f}",
                    f"{m.get('lh_precision', 0):.4f}",
                    f"{m.get('lh_recall', 0):.4f}",
                    f"{m.get('rh_precision', 0):.4f}",
                    f"{m.get('rh_recall', 0):.4f}",
                    self.args.folder,
                    self.args.ckpt_num
                ])
                
                # Plot results
                plot_name = f"plot_traj_{self.traj_idx}.jpg"
                plot_path = os.path.join(self.results_dir, plot_name)
                self.wrapper.plot_notes(plot_path)


            self.traj_idx += 1
            if self.traj_idx >= self.num_trajectories:
                self.get_logger().info("All trajectories completed. Shutting down.")
                self.wrapper.shutdown()
                return
            else:
                self.get_logger().info(f"Starting next trajectory ({self.traj_idx + 1}/{self.num_trajectories})...")
                self.wrapper.reset()
                self.wrapper.env.go_home()
                self.apply_offset_to_targets(self.args.left_offsets[self.traj_idx], self.args.right_offsets[self.traj_idx])
                self.init_robot(first=False)
                self.obs = self.wrapper.reset()
                return

        # 1. Inference
        with torch.no_grad():
            obs_torch = torch.from_numpy(self.obs).float().to(self.args.device).unsqueeze(0)
            action = self.agent.actor_mean(obs_torch).cpu().numpy().squeeze(0)
        
        # Record obs and action for debugging
        self.all_obs.append(self.obs.copy())
        self.all_actions.append(action.copy())

        # 2. Action Handling
        act_left_hand = action[0:15]
        act_right_hand = action[15:30]
        act_left_hand = self.wrapper.clip_and_scale_action(act_left_hand, -0.1, 0.1)
        act_right_hand = self.wrapper.clip_and_scale_action(act_right_hand, -0.1, 0.1)
        
        # 3. Integrate Deltas
        idx_r = min(self.wrapper.song_t_idx, len(self.target_ee_pose_arm_right) - 1)
        target_right_arm_pos = self.target_ee_pose_arm_right[idx_r][:3]
        target_right_arm_euler = self.default_ee_euler_r

        idx_l = min(self.wrapper.song_t_idx, len(self.target_ee_pose_arm_left) - 1)
        target_left_arm_pos = self.target_ee_pose_arm_left[idx_l][:3]
        target_left_arm_euler = self.default_ee_euler_l

        # Map 15-dim actions to 20-dim hands using indices from wrapper
        indices = self.wrapper.hand_indices
        target_right_hand = self.wrapper.get_right_hand_proprio().copy()
        target_right_hand[indices] += act_right_hand
        target_right_hand[self.wrapper.fixed_indices] = 1
        
        target_left_hand = self.wrapper.get_left_hand_proprio().copy()
        target_left_hand[indices] += act_left_hand
        target_left_hand[self.wrapper.fixed_indices] = 1

        target_right_hand = np.clip(target_right_hand, -0.3, RIGHT_JOINT_LIMITS[:, 1])
        target_left_hand = np.clip(target_left_hand, -0.3, LEFT_JOINT_LIMITS[:, 1])
        
        # 4. Step Environment
        self.obs, reward, done, info = self.wrapper.step(
            right_arm=(target_right_arm_pos, target_right_arm_euler),
            right_hand=target_right_hand,
            left_arm=(target_left_arm_pos, target_left_arm_euler),
            left_hand=target_left_hand,
            sim_action=action
        )

        if done:
            self.resetting = True

    def save_data(self):
        if len(self.all_obs) > 0:
            obs_array = np.array(self.all_obs)
            act_array = np.array(self.all_actions)
            obs_save_path = os.path.join(self.results_dir, "all_obs.npy")
            act_save_path = os.path.join(self.results_dir, "all_actions.npy")
            np.save(obs_save_path, obs_array)
            np.save(act_save_path, act_array)
            self.get_logger().info(f"Saved {len(obs_array)} observations/actions to {self.results_dir}")


    def shutdown(self):
        self.get_logger().info("Shutting down...")

        print("plot")
        self.wrapper.plot(self.results_dir)
        if hasattr(self.wrapper, 'shutdown'):
             self.wrapper.shutdown()
        
        exit()


def main():
    rclpy.init()
    node = Sim2RealInference()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_data()
        node.wrapper.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

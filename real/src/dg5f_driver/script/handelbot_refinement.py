import os
import time
import traceback
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pyrallis
import rclpy
import yaml
import yourdfpy
from rclpy.node import Node

from envs.piano.music.constants import is_white_key
from real.src.dg5f_driver.utils.bidex_env import BiDexEnv, BiDexEnvConfig, RealPianoSeparateWrapper
from real.src.dg5f_driver.utils.safe_utils import BimanualCollisionChecker
from real.src.dg5f_driver.utils.constants import RIGHT_JOINT_LIMITS, LEFT_JOINT_LIMITS

def white_key_distance(k1, k2):
    # Returns number of white keys between k1 and k2
    direction = 1 if k2 > k1 else -1
    count = 0
    curr = k1
    while curr != k2:
        curr += direction
        if is_white_key(curr):
            count += 1
    return count * direction
    
def white_key_dist(k1, k2):
    return abs(white_key_distance(k1, k2))


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


class HandelbotRefinement(Node):
    def __init__(self):
        super().__init__('handelbot_refinement')

        # Load configs
        config_path = os.path.join(os.path.dirname(__file__), "../config/handelbot_refinement.yaml")
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        self.args = SimpleNamespace(**config_data)
        if isinstance(self.args.action_high, list):
            self.args.action_high = np.array(self.args.action_high)
        run_dir = f"{self.args.rollout_root}/{self.args.folder}/optimize/"
        if os.path.exists(run_dir):
             i = 1
             while os.path.exists(f"{self.args.rollout_root}/{self.args.folder}/optimize{i}/"):
                 i += 1
             run_dir = f"{self.args.rollout_root}/{self.args.folder}/optimize{i}/"
        os.makedirs(run_dir, exist_ok=True)
        self.cfg = DataConfig(
            run_dir=run_dir,
            root=self.args.root,
            left_folder=self.args.left_folder,
            right_folder=self.args.right_folder
        )
        self.track_out_dir = os.path.join(run_dir, "tracking")
        self.get_logger().info(f"Root path: {self.cfg.root}")
        self.get_logger().info(f"Run Dir: {self.cfg.run_dir}")

        # Save config
        if not os.path.exists(self.track_out_dir):
            os.makedirs(self.track_out_dir, exist_ok=True)
        with open(os.path.join(self.cfg.run_dir, "config.yaml"), "w") as f:
            yaml.dump(config_data, f)
        self.get_logger().info(f"Config saved to {os.path.join(self.cfg.run_dir, 'config.yaml')}")

        # Load Environment
        bidex_cfg = pyrallis.load(BiDexEnvConfig, open(self.cfg.bidex_cfg_path, "r"))
        self.env = RealPianoSeparateWrapper(
            BiDexEnv(bidex_cfg, self), 
            sheet_music_path=self.args.sheet_music, 
            device=self.args.device,
        )
        self.setup_collision_checker()
        self.setup_trajectories(bidex_cfg)
        self.data_len = min(len(self.target_qpos_hand_left), len(self.target_qpos_hand_right))
        self.env.data_len = self.data_len

        # Optimization Hyperparameters
        self.subchunk_len = 2
        self.anneal_rate = 0.94
        self.guidance_check_start_offset = 0
        self.guidance_check_end_offset = 6
        self.reward_check_start_offset = 0 
        self.reward_check_end_offset = 6
        
        # Optimization State
        self.iteration = 0
        self.anneal_factor = 1.0
        num_chunks = (self.data_len + self.subchunk_len - 1) // self.subchunk_len + 1
        self.anneal_factors_left = np.ones(num_chunks)
        self.anneal_factors_right = np.ones(num_chunks)
        self.current_guided_residuals_left = np.zeros((self.data_len, self.args.action_dim // 2))
        self.current_guided_residuals_right = np.zeros((self.data_len, self.args.action_dim // 2))
        self.prev_guided_residuals_left = np.zeros((self.data_len, self.args.action_dim // 2))
        self.prev_guided_residuals_right = np.zeros((self.data_len, self.args.action_dim // 2))
        self.last_episode_pressed_keys = []
        self.best_f1 = -1.0
        
        # Execution State
        self.is_running_rollout = True
        self.in_warmup = True
        self.rollout_step = 0
        self.current_episode_rewards_left = []
        self.current_episode_rewards_right = []
        self.warmup_start_time = None
        self.stop_keyboard_thread = False
        
        input("Moving the robot to the start position! Press any key to continue, or Ctrl+C to stop.")
        self.init(first=True)
        
        print(f"Starting Optimization Loop. Total steps: {self.data_len}")
        self.timer = self.create_timer(1 / bidex_cfg.policy_freq, self.control_loop)


    def restore_terminal(self):
        if hasattr(self, 'old_term_settings'):
            termios.tcsetattr(self.stdin_fd, termios.TCSADRAIN, self.old_term_settings)


    def init(self, first=False):
        self.env.reset()
        
        # Initial States
        init_hand_r = self.target_qpos_hand_right[0]
        init_ee_r = self.target_ee_pose_arm_right[0]
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
            self.default_ee_euler_r = proprio_r.eef_euler
            proprio_l = self.env.get_left_arm_proprio()
            self.default_ee_euler_l = proprio_l.eef_euler
        self.env.right_move_to(ee_pos_r, self.default_ee_euler_r, 10)
        self.env.left_move_to(ee_pos_l, self.default_ee_euler_l, 10)

    def setup_collision_checker(self):
        right_urdf = yourdfpy.URDF.load(self.cfg.right_urdf_path)
        left_urdf = yourdfpy.URDF.load(self.cfg.left_urdf_path) 
        self.collision_checker = BimanualCollisionChecker(
            right_urdf=right_urdf,
            left_urdf=left_urdf,
        )

    def setup_trajectories(self, bidex_cfg):
        song_scale = max(1, 10 // bidex_cfg.policy_freq)
        self.target_qpos_hand_right = np.load(self.cfg.open_loop_right_hand_path)[:-1][::song_scale]
        self.target_ee_pose_arm_right = np.load(self.cfg.open_loop_right_arm_path)[:-1][::song_scale]
        self.target_qpos_hand_left = np.load(self.cfg.open_loop_left_hand_path)[:-1][::song_scale]
        self.target_ee_pose_arm_left = np.load(self.cfg.open_loop_left_arm_path)[:-1][::song_scale]

        # Standardize hand state from 15 to 20 DoF by adding fixed 4th joint value (1.0)
        if self.target_qpos_hand_right.shape[1] == 15:
            N = self.target_qpos_hand_right.shape[0]
            hand_reshaped = self.target_qpos_hand_right.reshape(N, 5, 3)
            ones = np.ones((N, 5, 1), dtype=self.target_qpos_hand_right.dtype)
            hand_concat = np.concatenate([hand_reshaped, ones], axis=2)
            self.target_qpos_hand_right = hand_concat.reshape(N, 20)

        if self.target_qpos_hand_left.shape[1] == 15:
            N = self.target_qpos_hand_left.shape[0]
            hand_reshaped = self.target_qpos_hand_left.reshape(N, 5, 3)
            ones = np.ones((N, 5, 1), dtype=self.target_qpos_hand_left.dtype)
            hand_concat = np.concatenate([hand_reshaped, ones], axis=2)
            self.target_qpos_hand_left = hand_concat.reshape(N, 20)

        # Adjust positions (same as actor)
        REAL_X_L, REAL_Y_L, REAL_Z_L = self.args.starting_pos_l
        REAL_X_R, REAL_Y_R, REAL_Z_R = self.args.starting_pos_r

        self.target_ee_pose_arm_right[:, 0] -= self.target_ee_pose_arm_right[0, 0]
        self.target_ee_pose_arm_left[:, 0] -= self.target_ee_pose_arm_left[0, 0]
        self.target_ee_pose_arm_right[:, 0] += REAL_X_R
        self.target_ee_pose_arm_left[:, 0] += REAL_X_L

        self.target_ee_pose_arm_right[:, 2] = REAL_Z_R - 0.01
        self.target_ee_pose_arm_left[:, 2] = REAL_Z_L

        self.target_ee_pose_arm_right[:, 1] += REAL_Y_R
        self.target_ee_pose_arm_left[:, 1] += REAL_Y_L

    def get_pressed_indices_for_finger(self, pressed_indices, target_notes, f_id):
        if len(pressed_indices) == 0:
            return np.array([])
            
        is_right_hand = f_id <= 5
        # Rule: Segregate by hand half
        # Middle C is around index 39 in 0-87 scale
        split_point = 39 
        if is_right_hand:
            hand_pressed = pressed_indices[pressed_indices >= split_point]
            hand_targets = [n for n in target_notes if n.fingering <= 5]
            sort_reverse = False
        else:
            hand_pressed = pressed_indices[pressed_indices < split_point]
            hand_targets = [n for n in target_notes if n.fingering > 5]
            sort_reverse = True
            
        if len(hand_pressed) == 0 or len(hand_targets) == 0:
            return np.array([])

        # Rule: If only one target note for the hand, all pressed belong to it
        if len(hand_targets) == 1:
            if hand_targets[0].fingering == f_id:
                return hand_pressed
            else:
                return np.array([])

        # Rule: Group into chunks
        hand_pressed = np.sort(hand_pressed)
        chunks = []
        current_chunk = [hand_pressed[0]]
        for i in range(1, len(hand_pressed)):
            if white_key_dist(hand_pressed[i-1], hand_pressed[i]) <= 1:
                current_chunk.append(hand_pressed[i])
            else:
                chunks.append(np.array(current_chunk))
                current_chunk = [hand_pressed[i]]
        chunks.append(np.array(current_chunk))
        
        # Rule: Assign fingers to chunks
        target_f_ids = sorted(list(set(n.fingering for n in hand_targets)), reverse=sort_reverse)
        
        if f_id not in target_f_ids:
            return np.array([])
            
        finger_idx = target_f_ids.index(f_id)
        if finger_idx < len(chunks):
            return chunks[finger_idx]
        
        return np.array([])

    def calculate_guided_residuals(self):
        # Calculate residuals for the NEXT rollout based on the LAST rollout's pressed keys
        self.prev_guided_residuals_left = self.current_guided_residuals_left.copy()
        self.prev_guided_residuals_right = self.current_guided_residuals_right.copy()

        # Reset current guided residuals
        self.current_guided_residuals_left.fill(0)
        self.current_guided_residuals_right.fill(0)
        
        if not self.last_episode_pressed_keys:
            return # First iteration, no guidance yet

        if self.args.use_chunk_annealing:
             print(f"Calculating guided residuals (Anneal mean: {np.mean(self.anneal_factors_left):.4f} / {np.mean(self.anneal_factors_right):.4f})")
        else:
             print(f"Calculating guided residuals (Anneal factor: {self.anneal_factor:.4f})")
        
        # Mappings
        rh_fingers = [(1, 0), (2, 1), (3, 2)] 
        lh_fingers = [(6, 0), (7, 1), (8, 2)]

        # Iterate through timeline
        for t in range(0, self.data_len, self.subchunk_len):
            subtraj_len = min(self.subchunk_len, self.data_len - t)
            
            # Check window [t + start, t + end] for errors
            check_start = max(0, t + self.guidance_check_start_offset)
            check_end = min(self.data_len, t + self.guidance_check_end_offset)
            
            # Helper to calculate delta for a hand
            def get_delta_for_hand(fingers, is_right_hand=True):
                hand_delta = np.zeros(self.args.action_dim // 2)
                
                for f_id, a_idx in fingers:
                    finger_deltas = []
                    
                    # Check errors in previous window
                    for check_t in range(check_start, check_end):
                        if check_t >= len(self.last_episode_pressed_keys): break
                        
                        target_notes = self.env._notes[check_t]
                        pressed_mask = self.last_episode_pressed_keys[check_t]
                        pressed_indices = np.where(pressed_mask > 0)[0]
                        
                        target_note = next((n for n in target_notes if n.fingering == f_id), None)
                        if target_note is None: continue
                        
                        target_key = target_note.key
                        
                        pressed_indices_for_f = self.get_pressed_indices_for_finger(pressed_indices, target_notes, f_id)
                        if len(pressed_indices_for_f) == 0:
                            continue # Missed completely or nothing pressed for this finger
                            
                        dists = np.abs(pressed_indices_for_f - target_key)
                        if np.min(dists) > 6:
                            continue # Too far
                            
                        closest_key = pressed_indices_for_f[np.argmin(dists)]
                        step_delta = 0
                        
                        if self.args.use_chunk_annealing:
                            current_anneal = self.anneal_factors_right[t // self.subchunk_len] if is_right_hand else self.anneal_factors_left[t // self.subchunk_len]
                        else:
                            current_anneal = self.anneal_factor
                        
                        if closest_key < target_key:
                            step_delta = 0.3 * current_anneal # * white_key_dist
                        elif closest_key > target_key:
                            step_delta = -0.3 * current_anneal # * white_key_dist
                        else:
                            # Hit correct, check neighbors
                            if target_key + 1 in pressed_indices_for_f or target_key + 2 in pressed_indices_for_f:
                                step_delta = -0.15 * current_anneal
                            elif target_key - 1 in pressed_indices_for_f or target_key - 2 in pressed_indices_for_f:
                                step_delta = 0.15 * current_anneal
                        
                        if step_delta != 0:
                            finger_deltas.append(step_delta) 

                    # Apply to 0th joint (index 0, 3, 6 within the 9-dim action)
                    if finger_deltas:
                        finger_deltas_mean = np.mean(finger_deltas)
                        hand_delta[a_idx * 3] = finger_deltas_mean
                        if finger_deltas_mean > 0:
                            # move the finger to the right more to the right
                            hand_delta[(a_idx + 1) * 3] = finger_deltas_mean * 0.3
                        if finger_deltas_mean < 0 and a_idx > 0:
                            # move finger to the left more to the left
                            hand_delta[(a_idx - 1) * 3] = finger_deltas_mean * 0.3
                    
                return hand_delta

            delta_l = get_delta_for_hand(lh_fingers, is_right_hand=False)
            delta_r = get_delta_for_hand(rh_fingers, is_right_hand=True)
            
            # Momentum Damping
            if self.args.use_momentum_damping:
                prev_delta_l = self.prev_guided_residuals_left[t]
                prev_delta_r = self.prev_guided_residuals_right[t]
                
                # Sign flip -> strong damping
                mask_l = np.sign(delta_l) != np.sign(prev_delta_l)
                delta_l[mask_l] *= 0.7
                
                mask_r = np.sign(delta_r) != np.sign(prev_delta_r)
                delta_r[mask_r] *= 0.7
            
            # Apply same delta to all steps in subtrajectory [t, t+subtraj_len]
            for i in range(subtraj_len):
                self.current_guided_residuals_left[t + i] = delta_l
                self.current_guided_residuals_right[t + i] = delta_r


    def update_best_residuals(self):
        current_rewards_l = np.array(self.current_episode_rewards_left)
        current_rewards_r = np.array(self.current_episode_rewards_right)

        updates_count = 0
        
        for t in range(0, self.data_len, self.subchunk_len):
            # Define reward window
            r_start = max(0, t + self.reward_check_start_offset)
            r_end = min(self.data_len, t + self.reward_check_end_offset) # t+4 inclusive
            
            # Calculate mean reward for this window
            curr_mean_l = np.mean(current_rewards_l[r_start:r_end])
            best_mean_l = np.mean(self.best_rewards_left[r_start:r_end])
            
            if curr_mean_l >= best_mean_l:
                # Update best for this chunk
                sub_len = min(self.subchunk_len, self.data_len - t)
                for i in range(sub_len):
                    idx = t + i
                    self.best_residuals_left[idx] = self.best_residuals_left[idx] + self.current_guided_residuals_left[idx]
                    self.best_rewards_left[idx] = current_rewards_l[idx]
                self.best_rewards_left[r_start:r_end] = current_rewards_l[r_start:r_end]
                updates_count += 1
                if self.args.use_chunk_annealing:
                    self.anneal_factors_left[t // self.subchunk_len] *= 0.7
            else:
                if self.args.use_chunk_annealing:
                    self.anneal_factors_left[t // self.subchunk_len] *= 0.97

            # Right hand
            curr_mean_r = np.mean(current_rewards_r[r_start:r_end])
            best_mean_r = np.mean(self.best_rewards_right[r_start:r_end])
            if curr_mean_r >= best_mean_r:
                sub_len = min(self.subchunk_len, self.data_len - t)
                for i in range(sub_len):
                    idx = t + i
                    self.best_residuals_right[idx] = self.best_residuals_right[idx] + self.current_guided_residuals_right[idx]

                self.best_rewards_right[r_start:r_end] = current_rewards_r[r_start:r_end]
                updates_count += 1
                if self.args.use_chunk_annealing:
                    self.anneal_factors_right[t // self.subchunk_len] *= 0.7
            else:
                if self.args.use_chunk_annealing:
                    self.anneal_factors_right[t // self.subchunk_len] *= 0.97

        # Clip anneal factors
        self.anneal_factors_left = np.clip(self.anneal_factors_left, 0.1, 1.5)
        self.anneal_factors_right = np.clip(self.anneal_factors_right, 0.1, 1.5)

        self.best_residuals_left = np.clip(self.best_residuals_left, -self.args.action_high, self.args.action_high)
        self.best_residuals_right = np.clip(self.best_residuals_right, -self.args.action_high, self.args.action_high)
                
        print(f"Updated best residuals for {updates_count} chunks.")


    def save_optimized_trajectory(self, f1_score=None, res_left=None, res_right=None):
        final_l = self.target_qpos_hand_left.copy()
        final_r = self.target_qpos_hand_right.copy()
        
        # Apply residuals
        if res_left is None:
            res_left = self.current_guided_residuals_left
            res_right = self.current_guided_residuals_right
        res_left = np.clip(res_left, -self.args.action_high, self.args.action_high)
        res_right = np.clip(res_right, -self.args.action_high, self.args.action_high)
        for t in range(self.data_len):
            final_l[t, self.args.action_idxs] += res_left[t]
            final_r[t, self.args.action_idxs] += res_right[t]
            
        final_l = np.clip(final_l, -0.3, LEFT_JOINT_LIMITS[:, 1])
        final_r = np.clip(final_r, -0.3, RIGHT_JOINT_LIMITS[:, 1])
        
        if f1_score is None:
            np.save(os.path.join(self.cfg.run_dir, f"optimized_residuals_left_final.npy"), self.current_guided_residuals_left)
            np.save(os.path.join(self.cfg.run_dir, f"optimized_residuals_right_final.npy"), self.current_guided_residuals_right)
            np.save(os.path.join(self.cfg.run_dir, f"optimized_traj_left.npy"), final_l)
            np.save(os.path.join(self.cfg.run_dir, f"optimized_traj_right.npy"), final_r)
        else: 
            np.save(os.path.join(self.cfg.run_dir, f"optimized_residuals_left_f1.npy"), self.current_guided_residuals_left)
            np.save(os.path.join(self.cfg.run_dir, f"optimized_residuals_right_f1.npy"), self.current_guided_residuals_right)
            np.save(os.path.join(self.cfg.run_dir, f"optimized_traj_left_f1.npy"), final_l)
            np.save(os.path.join(self.cfg.run_dir, f"optimized_traj_right_f1.npy"), final_r)
            with open(os.path.join(self.cfg.run_dir, "best_f1_score.txt"), "w") as f:
                 f.write(str(f1_score))

        plot_name = f"optimized_plot_f1_{f1_score:.4f}.jpg" if f1_score is not None else "optimized_plot.jpg"
        plot_path = os.path.join(self.cfg.run_dir, plot_name)
        if hasattr(self.env, 'plot_notes'):
             self.env.plot_notes(plot_path)

        print("Saved optimized trajectories, score, and plot.")


    def control_loop(self):
        if not self.is_running_rollout:
            return

        if self.in_warmup:
            current_time = self.get_clock().now().nanoseconds
            if self.warmup_start_time is None:
                self.warmup_start_time = current_time
                self.get_logger().info("Starting warmup...")
            
            elapsed = (current_time - self.warmup_start_time) / 1e9
            if elapsed < self.args.warmup_duration_s:
                return
            
            self.in_warmup = False
            self.get_logger().info("Warmup complete. Starting rollout...")

        if self.rollout_step >= self.data_len:
            self.is_running_rollout = False
            self.last_episode_pressed_keys = list(self.env.music_history['pressed'])
            
            # Update anneal factor
            if self.iteration > 0:
                if not self.args.use_chunk_annealing:
                    self.anneal_factor = np.exp(-0.05 * (self.iteration - 1))
            else:
                self.rewards_left = np.array(self.current_episode_rewards_left)
                self.rewards_right = np.array(self.current_episode_rewards_right)
            
            # Check F1 and Save
            if hasattr(self.env, 'episode_metrics'):
                f1 = self.env.episode_metrics.get('f1', 0.0)
                print(f"Episode F1: {f1:.4f}")
                if f1 > self.best_f1:
                    print(f"New best F1: {f1:.4f} (was {self.best_f1:.4f}). Saving...")
                    self.best_f1 = f1
                    self.save_optimized_trajectory(f1_score=f1, res_left=self.current_guided_residuals_left, res_right=self.current_guided_residuals_right)
            
            self.iteration += 1
            self.calculate_guided_residuals()
            self.env.env.go_home()
            self.init()
            self.in_warmup = True
            self.warmup_start_time = None
            self.rollout_step = 0
            self.current_episode_rewards_left = []
            self.current_episode_rewards_right = []
            self.is_running_rollout = True
            
            print(f"Starting Iteration {self.iteration}")
            if self.iteration > self.args.max_iterations: # Max iterations
                self.save_optimized_trajectory()
                self.shutdown()
            return

        # 1. Get Base Targets
        target_hand_r = self.target_qpos_hand_right[self.rollout_step].copy()
        target_hand_l = self.target_qpos_hand_left[self.rollout_step].copy()
        ee_pos_r = self.target_ee_pose_arm_right[self.rollout_step][:3]
        ee_pos_l = self.target_ee_pose_arm_left[self.rollout_step][:3]
        
        # 2. Add Residuals (Guided)
        res_l = self.current_guided_residuals_left[self.rollout_step]
        res_r = self.current_guided_residuals_right[self.rollout_step]
        res_l = np.clip(res_l, -self.args.action_high, self.args.action_high)
        res_r = np.clip(res_r, -self.args.action_high, self.args.action_high)
        
        target_hand_r[self.args.action_idxs] += res_r
        target_hand_l[self.args.action_idxs] += res_l
        target_hand_r = np.clip(target_hand_r, -0.3, RIGHT_JOINT_LIMITS[:, 1])
        target_hand_l = np.clip(target_hand_l, -0.3, LEFT_JOINT_LIMITS[:, 1])

        # 3. Safety Check
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
        
        target_hand_r, target_hand_l = self.collision_checker.get_safe_move(
            target_hand_r, target_hand_l,
            curr_r, curr_l
        )

        # 4. Step
        _, (rew_left, rew_right), _, _ = self.env.step(
            right_arm=(ee_pos_r, self.default_ee_euler_r),
            left_arm=(ee_pos_l, self.default_ee_euler_l),
            right_hand=target_hand_r,
            left_hand=target_hand_l,
        )
        
        self.current_episode_rewards_left.append(rew_left)
        self.current_episode_rewards_right.append(rew_right)
        self.rollout_step += 1


    def shutdown(self):
        self.get_logger().info("Shutting down...")
        self.stop_keyboard_thread = True
        self.restore_terminal()
        if self.env:
            self.env.shutdown()
        exit()

def main(args=None):
    rclpy.init(args=args)
    node = HandelbotRefinement()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
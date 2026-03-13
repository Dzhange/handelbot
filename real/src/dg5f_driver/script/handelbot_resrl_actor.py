import copy
import csv
import os
import select
import sys
import termios
import threading
import time
import traceback
import tty
from collections import deque
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np
import pyrallis
import rclpy
import torch
import yaml
import yourdfpy
import zmq
from rclpy.node import Node
from torch.utils.tensorboard import SummaryWriter

from envs.piano.music.constants import is_white_key
from real.src.dg5f_driver.utils.bidex_env import BiDexEnv, BiDexEnvConfig, RealPianoSeparateWrapper
from real.src.dg5f_driver.utils.rl.td3 import Actor, Logger
from real.src.dg5f_driver.utils.safe_utils import BimanualCollisionChecker
from real.src.dg5f_driver.utils.constants import RIGHT_JOINT_LIMITS, LEFT_JOINT_LIMITS


def white_key_distance(k1, k2):
    """Returns the number of white keys between k1 and k2."""
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


class stages:
    DO_NOTHING = 0
    DO_WARMUP = 1
    DO_CONTROL_LOOP = 2
    RESETTING = 3
    MANUAL_RESET = 4


class policy_stages:
    SAMPLE = 0
    EVAL = 1
    GUIDED_NOISE = 2

@dataclass(frozen=True)
class DataConfig:
    root: str
    run_dir: str
    left_folder: str
    right_folder: str
    left_hand_folder: str
    right_hand_folder: str

    @property
    def right_urdf_path(self):
        return os.path.join(self.root, "assets/tesollo_delto/urdf/dg5f_right_nothumb.urdf")

    @property
    def left_urdf_path(self):
        return os.path.join(self.root, "assets/tesollo_delto/urdf/dg5f_left_nothumb.urdf")

    @property
    def open_loop_right_hand_path(self):
        return f"{self.root}/real/ckpts/{self.right_hand_folder}/optimized_traj_right_f1.npy"

    @property
    def open_loop_left_hand_path(self):
        return f"{self.root}/real/ckpts/{self.left_hand_folder}/optimized_traj_left_f1.npy"

    @property
    def open_loop_right_arm_path(self):
        return f"{self.root}/real/ckpts/{self.right_folder}/right_arm_target_ee_pose.npy"

    @property
    def open_loop_left_arm_path(self):
        return f"{self.root}/real/ckpts/{self.left_folder}/left_arm_target_ee_pose.npy"

    @property
    def bidex_cfg_path(self):
        return os.path.join(self.root, "real/src/dg5f_driver/config/handelbot_resrl.yaml")


class HandelbotResRLActor(Node):
    def __init__(self):
        super().__init__('handelbot_resrl_actor')

        # Load configs
        config_path = os.path.join(os.path.dirname(__file__), "../config/residual_rl_separate.yaml")
        with open(config_path, "r") as f:
            self.config_dict = yaml.safe_load(f)
        self.args = SimpleNamespace(**self.config_dict)
        run_dir = os.path.join(self.args.rollout_root, self.args.folder)
        os.makedirs(run_dir, exist_ok=True)
        self.cfg = DataConfig(
            root=self.args.root,
            run_dir=run_dir,
            left_folder=self.args.left_folder,
            right_folder=self.args.right_folder,
            left_hand_folder=self.args.left_hand_folder,
            right_hand_folder=self.args.right_hand_folder
        )
        self.track_out_dir = os.path.join(run_dir, "tracking")
        self.get_logger().info(f"Root path: {self.cfg.root}")
        self.get_logger().info(f"Run Dir: {self.cfg.run_dir}")

        # Save config
        if not os.path.exists(self.track_out_dir):
            os.makedirs(self.track_out_dir, exist_ok=True)
        with open(os.path.join(self.cfg.run_dir, "config.yaml"), "w") as f:
            yaml.dump(self.config_dict, f)
        self.get_logger().info(f"Config saved to {os.path.join(self.cfg.run_dir, 'config.yaml')}")

        # Load hyperparameters
        if isinstance(self.args.action_high, list):
            self.args.action_high = np.array(self.args.action_high)
        if isinstance(self.args.init_policy_noise, list):
            self.args.init_policy_noise = np.array(self.args.init_policy_noise)
        if isinstance(self.args.policy_noise, list):
            self.args.policy_noise = np.array(self.args.policy_noise)
        self.history_length = self.args.history_length
        self.get_logger().info(f"History Length: {self.history_length}")
        self.left_history = deque(maxlen=self.history_length)
        self.right_history = deque(maxlen=self.history_length)
        self.episode_residual_actions = []
        self.last_episode_residual_actions = []
        self.last_episode_pressed_keys = []

        # Load Environment
        input("Ready to start? Press any key to continue, or Ctrl+C to stop.")
        bidex_cfg = pyrallis.load(BiDexEnvConfig, open(self.cfg.bidex_cfg_path, "r"))
        self.env = RealPianoSeparateWrapper(
            BiDexEnv(bidex_cfg, self), 
            sheet_music_path=self.args.sheet_music, 
            device=self.args.buffer_device,
        )
        self.setup_collision_checker()
        self.setup_trajectories(bidex_cfg)

        # Replay Buffer
        self.batch_size = self.args.zmq_batch_size
        self.batch = []
        ctx = zmq.Context()
        self.sock = ctx.socket(zmq.PUSH)
        self.sock.connect("tcp://127.0.0.1:5588")
        
        # State Variables
        self.pure_noise_left = None
        self.pure_noise_right = None
        self.curr_ckpt = 0
        self.policy_stage = policy_stages.SAMPLE
        self.eval_ep_count = 0
        self.num_eval_episodes = 5

        # Actors
        self.actor_left = Actor(
            obs_dim=self.args.obs_dim_left,
            action_dim=self.args.action_dim // 2,
            action_high=self.args.action_high,
            policy_noise=self.args.policy_noise,
            actor_hidden_dims=self.args.actor_hidden_dims,
            dropout=self.args.dropout,
            layernorm=self.args.layernorm,
            history_length=self.history_length,
            noise_beta=self.args.noise_beta
        ).to(self.args.buffer_device)
        self.actor_right = Actor(
            obs_dim=self.args.obs_dim_right,
            action_dim=self.args.action_dim // 2,
            action_high=self.args.action_high,
            policy_noise=self.args.policy_noise,
            actor_hidden_dims=self.args.actor_hidden_dims,
            dropout=self.args.dropout,
            layernorm=self.args.layernorm,
            history_length=self.history_length,
            noise_beta=self.args.noise_beta
        ).to(self.args.buffer_device)
        if self.args.checkpoint:
            self.get_logger().info(f"Loading checkpoint from {self.args.checkpoint}")
            ckpt = torch.load(self.args.checkpoint, map_location=self.args.buffer_device)
            self.actor_left.load_state_dict(ckpt['actor_left'])
            self.actor_right.load_state_dict(ckpt['actor_right'])
        else:
            self.get_logger().info("No checkpoint loaded, using random weights")

        # Logging
        self.actor_log_dir = f"{self.args.root}/{self.args.folder}/actor/"
        if os.path.exists(self.actor_log_dir):
            i = 1
            while os.path.exists(f"{self.args.root}/{self.args.folder}/actor{i}/"):
                i += 1
            try:
                if i == 1:
                    last_log_dir = f"{self.args.root}/{self.args.folder}/actor/"
                else:
                    last_log_dir = f"{self.args.root}/{self.args.folder}/actor{i-1}/"
                
                log_csv_path = os.path.join(last_log_dir, "log.csv")
                if os.path.exists(log_csv_path):
                    with open(log_csv_path, "r") as f:
                        last_line = deque(csv.reader(f), 1)[0]
                        if last_line and last_line[0] != 'step': # ensure not header
                             last_step = int(last_line[0])
                             self.global_idx = last_step + 1
                             self.get_logger().info(f"Resuming global_idx from {self.global_idx} (found in {log_csv_path})")
            except Exception as e:
                 self.get_logger().warn(f"Failed to resume global_idx: {e}")

            self.actor_log_dir = f"{self.args.root}/{self.args.folder}/actor{i}/"
        writer = SummaryWriter(self.actor_log_dir)
        self.logger = Logger(log_wandb=False, tensorboard=writer, log_dir=self.actor_log_dir)
        self.env.log_dir = self.actor_log_dir

        # Save config
        os.makedirs(self.actor_log_dir, exist_ok=True)
        combined_config = copy.deepcopy(self.config_dict)
        combined_config.update(asdict(self.cfg))
        with open(os.path.join(self.actor_log_dir, "config.yaml"), "w") as f:
            yaml.dump(combined_config, f)

        # Initialize robot
        self.init(first=True)
        print(f"Run: {self.cfg.left_hand_folder} {self.cfg.right_hand_folder}")
        input("Beginning open loop rollout! Press any key to continue, or Ctrl+C to stop.")

        # Timer
        self._start_keyboard_listener()
        self.warmup_start_time = None
        self.policy_timer = self.create_timer(1 / bidex_cfg.policy_freq, self.policy_timer_callback)
        self.start_time = self.get_clock().now().nanoseconds
        self.last_action_time = time.time()
        self.stage = stages.DO_WARMUP
        self.ckpt_timer = self.create_timer(20.0, self.check_new_checkpoint) # Check every 20 sec


    def _start_keyboard_listener(self):
        self.stop_keyboard_thread = False
        
        # Save terminal settings
        self.stdin_fd = sys.stdin.fileno()
        try:
            self.old_term_settings = termios.tcgetattr(self.stdin_fd)
            # Set to cbreak mode (non-blocking, no echo, but allows signals like Ctrl+C)
            tty.setcbreak(self.stdin_fd)
            self.keyboard_thread = threading.Thread(target=self._keyboard_listener_loop, daemon=True)
            self.keyboard_thread.start()
            self.get_logger().info("Keyboard listener started. Press Ctrl+E to reset env.")
        except Exception as e:
            self.get_logger().error(f"Failed to setup keyboard listener: {e}")

    def _keyboard_listener_loop(self):
        try:
            while not self.stop_keyboard_thread:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    key = sys.stdin.read(1)
                    if key == '\x05': # Ctrl+E for manual resets
                        self.stage = stages.MANUAL_RESET
                        print("\nCtrl+E detected! Resetting environment...\r")
        except Exception as e:
            pass

    def restore_terminal(self):
        if hasattr(self, 'old_term_settings'):
            termios.tcsetattr(self.stdin_fd, termios.TCSADRAIN, self.old_term_settings)

    def check_new_checkpoint(self):
        try:
             dir_path = f"{self.args.root}/{self.args.folder}"
             if not os.path.exists(dir_path): return

             ckpts = [f for f in os.listdir(dir_path) if f.startswith("ckpt_") and f.endswith(".pt")]
             if not ckpts:
                 return
             
             ckpts.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
             latest_ckpt = ckpts[-1]
             ckpt_path = os.path.join(dir_path, latest_ckpt)
             step_num = int(latest_ckpt.split("_")[1].split(".")[0])

             # Basic check to see if it's different/new (could store last loaded path)
             if not hasattr(self, 'last_loaded_ckpt') or self.curr_ckpt < step_num:
                 self.get_logger().info(f"Found new checkpoint: {latest_ckpt}. Loading...")
                 self.stage = stages.DO_NOTHING
                 
                 try:
                     # Load checkpoint
                     ckpt = torch.load(ckpt_path, map_location=self.args.buffer_device)
                     self.actor_left.load_state_dict(ckpt['actor_left'])
                     self.actor_right.load_state_dict(ckpt['actor_right'])
                     self.last_loaded_ckpt = ckpt_path
                     self.get_logger().info(f"Successfully loaded {latest_ckpt}")
                     
                     self.curr_ckpt = step_num
                     if hasattr(self.args, 'eval_ckpt_freq'):
                        for ckpt_file in ckpts: # All found checkpoints
                            try:
                                s_num = int(ckpt_file.split("_")[1].split(".")[0])
                                if s_num % self.args.eval_ckpt_freq != 0:
                                    full_p = os.path.join(dir_path, ckpt_file)
                                    if os.path.exists(full_p):
                                        os.remove(full_p)
                                        self.get_logger().info(f"Deleted checkpoint: {ckpt_file}")
                            except Exception as e:
                                pass

                 except Exception as e:
                     self.get_logger().error(f"Failed to load checkpoint {latest_ckpt}: {e}")
                 finally:
                     self.stage = stages.DO_CONTROL_LOOP 
        except Exception as e:
             self.get_logger().error(f"Error checking checkpoints: {e}")

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

        song_scale = max(1, 10 // bidex_cfg.policy_freq)
        self.target_qpos_hand_right = np.load(self.cfg.open_loop_right_hand_path)[:-1][::song_scale]
        self.target_ee_pose_arm_right = np.load(self.cfg.open_loop_right_arm_path)[:-1][::song_scale]

        self.target_qpos_hand_left = np.load(self.cfg.open_loop_left_hand_path)[:-1][::song_scale]
        self.target_ee_pose_arm_left = np.load(self.cfg.open_loop_left_arm_path)[:-1][::song_scale]

        if self.args.from_scratch:
            self.target_qpos_hand_left *= 0
            self.target_qpos_hand_right *= 0
            
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

        if self.target_qpos_hand_left.shape[1] == 16:
            self.target_qpos_hand_left = np.concatenate((np.zeros_like(self.target_qpos_hand_left[:, :4]), self.target_qpos_hand_left), axis=1)
        elif self.target_qpos_hand_left.shape[1] == 20:
            self.target_qpos_hand_left[:, :4] = 0
        else: 
            raise NotImplementedError
        if self.target_qpos_hand_right.shape[1] == 16:
            self.target_qpos_hand_right = np.concatenate((np.zeros_like(self.target_qpos_hand_right[:, :4]), self.target_qpos_hand_right), axis=1)
        elif self.target_qpos_hand_right.shape[1] == 20:
            self.target_qpos_hand_right[:, :4] = 0
        else:
            raise NotImplementedError

        if self.args.from_scratch:
            self.target_qpos_hand_left[17] = -0.5
            self.target_qpos_hand_right[17] = -0.5

        input("Moving the robot to the start position! Press any key to continue, or Ctrl+C to stop.")

        # Adjust offsets (same as actor)
        x_off_left, y_off_left, z_off_left = 0.022, 0.05, -0.018
        x_off_right, y_off_right, z_off_right = 0.03, 0.02, 0.027 

        REAL_X_R = 0.525 + x_off_right 
        REAL_Z_R = 0.27 + z_off_right

        REAL_X_L = 0.525 + x_off_left
        REAL_Z_L = 0.27 + z_off_left

        self.target_ee_pose_arm_right[:, 0] -= self.target_ee_pose_arm_right[0, 0]
        self.target_ee_pose_arm_left[:, 0] -= self.target_ee_pose_arm_left[0, 0]

        if "elise" in self.args.sheet_music:  
            self.target_ee_pose_arm_right[:, 0] *= 1.5
            self.target_ee_pose_arm_left[:, 0] *= 1.5
            self.target_ee_pose_arm_right[:, 2] += self.target_ee_pose_arm_right[:, 0] * 1.4
            self.target_ee_pose_arm_left[:, 2] += self.target_ee_pose_arm_left[:, 0] * 1.4

        self.target_ee_pose_arm_right[:, 0] += REAL_X_R
        self.target_ee_pose_arm_left[:, 0] += REAL_X_L

        self.target_ee_pose_arm_right[:, 2] = REAL_Z_R - 0.01
        self.target_ee_pose_arm_left[:, 2] = REAL_Z_L

        self.target_ee_pose_arm_right[:, 1] += (0.1 + y_off_right) 
        self.target_ee_pose_arm_left[:, 1] += (-0.1 + y_off_left) 

        self.data_len = min(len(self.target_qpos_hand_left), len(self.target_qpos_hand_right))
        self.env.data_len = self.data_len
        self.env.target_qpos_hand_left = self.target_qpos_hand_left
        self.env.target_qpos_hand_right = self.target_qpos_hand_right
        self.get_logger().info(f"Loaded open loop data with {self.data_len} steps")


    def _now_s(self):
        return (self.get_clock().now().nanoseconds - self.start_time) / 1e9

    def init(self, first=False):
        # Reset Env
        obs = self.env.reset()
        obs_left, obs_right = obs
        for _ in range(self.history_length):
            self.left_history.append(obs_left)
            self.right_history.append(obs_right)

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

        # Chunking state
        self.chunk_step = 0
        self.accum_rew_left = np.zeros(3)
        self.accum_rew_right = np.zeros(3)
        self.stored_action_left = np.zeros(self.args.action_dim // 2)
        self.stored_action_right = np.zeros(self.args.action_dim // 2)
        self.chunk_start_obs_left = None
        self.chunk_start_obs_right = None
        self.pure_noise_left = None
        self.pure_noise_right = None

    def warmup(self):
        current_time = self.get_clock().now().nanoseconds
        if self.warmup_start_time is None:
            self.warmup_start_time = current_time
            self.get_logger().info("Starting warmup...")

        elapsed = (current_time - self.warmup_start_time) / 1e9
        if elapsed < self.args.warmup_duration_s:
            return 

        self.stage = stages.DO_CONTROL_LOOP
        self.get_logger().info("Warmup complete. Starting (open) control loop...")

    def shutdown(self):
        self.get_logger().info("Shutting down...")
        try:
            if hasattr(self, 'sock'):
                self.sock.send_pyobj("SHUTDOWN")
                self.get_logger().info("Sent SHUTDOWN signal to learner.")
        except Exception as e:
            self.get_logger().error(f"Failed to send SHUTDOWN signal: {e}")

        if self.env:
            self.env.plot(self.actor_log_dir)
            if hasattr(self.env, 'shutdown'):
                 self.env.shutdown()
        
        self.stop_keyboard_thread = True
        self.restore_terminal()
        exit()

    def policy_timer_callback(self):
        if self.stage == stages.MANUAL_RESET:
            self.stage = stages.DO_NOTHING
            self.get_logger().warn("MANUAL RESET TRIGGERED (Ctrl+E)...")
            try:
                self.env.env.go_home()
                self.init()
            except Exception as e:
                self.get_logger().error(f"Error during manual reset: {e}")
            self.batch[-1]['done'] = True
            self.batch[-1]['mask'] = 0
            self.stage = stages.DO_WARMUP
        elif self.stage == stages.DO_WARMUP:
            self.warmup()
        elif self.stage == stages.DO_NOTHING:
            return
        elif self.stage == stages.RESETTING:
            print("episode", self.global_ep)
            if self.policy_stage == policy_stages.EVAL:
                self.eval_ep_count += 1
                if self.eval_ep_count >= self.num_eval_episodes:
                    self.policy_stage = policy_stages.SAMPLE
                    self.eval_ep_count = 0
                    print("EVALUATION DONE")
                    self.sock.send_pyobj("RESUME")
                else:
                    print(f"RUNNING EVALUATION TRAJECTORY {self.eval_ep_count + 1}/{self.num_eval_episodes}!")
            else:
                 self.policy_stage = policy_stages.SAMPLE
                 
            if self.args.guided_noise and self.policy_stage == policy_stages.SAMPLE:
                self.policy_stage = policy_stages.GUIDED_NOISE

            if (self.global_ep == 25 or self.global_ep % 20 == 2) and self.global_ep > 23:
                self.policy_stage = policy_stages.EVAL
                self.eval_ep_count = 0
                print(f"STARTING EVALUATION TRAJECTORIES! (1/{self.num_eval_episodes})")
                self.sock.send_pyobj("PAUSE")
            
            self.env.global_idx = self.global_idx
            self.env.is_eval = (self.policy_stage == policy_stages.EVAL)

            self.stage = stages.DO_NOTHING
            if self.global_ep % self.args.full_reset_freq == 0:
                print("Full reset of environment")
                self.sock.send_pyobj("PAUSE")
                self.env.env.go_home()
                self.init()
                if self.policy_stage != policy_stages.EVAL:
                    self.sock.send_pyobj("RESUME")
            else:
                self.init()
            self.stage = stages.DO_WARMUP
        elif self.stage == stages.DO_CONTROL_LOOP:
            self.policy_loop()
        else:
            raise ValueError(f"Invalid stage: {self.stage}")

    def policy_loop(self):
        s0 = time.time()

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

        # 2. Residual Policy Inference
        obs_left_stacked = np.concatenate(list(self.left_history))
        obs_right_stacked = np.concatenate(list(self.right_history))
        
        obs_left_tensor = torch.from_numpy(obs_left_stacked).float().to(self.args.buffer_device).unsqueeze(0) # Add batch dim
        obs_right_tensor = torch.from_numpy(obs_right_stacked).float().to(self.args.buffer_device).unsqueeze(0)

        # (b) get actions (only if new chunk)
        if self.chunk_step == 0:
            self.chunk_start_obs_left = obs_left_stacked
            self.chunk_start_obs_right = obs_right_stacked
            
            if self.global_ep == 0 and not self.args.from_scratch:
                 action_left, action_right = 0, 0
            elif self.policy_stage == policy_stages.EVAL:
                with torch.no_grad():
                    action_left = self.actor_left.get_eval_action(obs_left_tensor)
                    action_right = self.actor_right.get_eval_action(obs_right_tensor)
                action_left = action_left.cpu().numpy().flatten()
                action_right = action_right.cpu().numpy().flatten()
            elif self.policy_stage == policy_stages.GUIDED_NOISE:
                if np.random.random() < self.args.guided_noise_prob:
                    guidance_left, guidance_right = self.get_guidance_sign(self.env.song_t_idx)
                else:
                    guidance_left = torch.zeros(self.args.action_dim // 2)
                    guidance_right = torch.zeros(self.args.action_dim // 2)
                
                # Use same policy noise logic as SAMPLE
                if self.global_idx < self.args.n_policy_noise_warmup_steps:
                    ratio = self.global_idx / self.args.n_policy_noise_warmup_steps
                    policy_noise = ratio * self.args.policy_noise + (1 - ratio) * self.args.init_policy_noise
                else:
                    policy_noise = None
                    
                with torch.no_grad():
                    action_left, _, _ = self.actor_left.get_guided_action(obs_left_tensor, guidance_left, policy_noise)
                    action_right, _, _ = self.actor_right.get_guided_action(obs_right_tensor, guidance_right, policy_noise)
                
                action_left = action_left.cpu().numpy().flatten()
                action_right = action_right.cpu().numpy().flatten()
            elif self.policy_stage == policy_stages.SAMPLE or ((self.global_idx < self.args.learning_starts) and self.policy_stage != policy_stages.EVAL):
                if self.global_idx < self.args.learning_starts:
                     if self.pure_noise_left is None:
                        self.pure_noise_left = np.random.randn(self.args.action_dim//2)
                        self.pure_noise_right = np.random.randn(self.args.action_dim//2)
                     else:
                        beta = self.args.noise_beta
                        self.pure_noise_left = beta * self.pure_noise_left + np.sqrt(1 - beta**2) * np.random.randn(self.args.action_dim//2)
                        self.pure_noise_right = beta * self.pure_noise_right + np.sqrt(1 - beta**2) * np.random.randn(self.args.action_dim//2)
    
                     action_left = self.pure_noise_left * self.args.action_high
                     action_right = self.pure_noise_right * self.args.action_high
                     action_left = np.clip(action_left, -self.args.action_high, self.args.action_high)
                     action_right = np.clip(action_right, -self.args.action_high, self.args.action_high)

                else:
                    if self.global_idx < self.args.n_policy_noise_warmup_steps:
                        ratio = self.global_idx / self.args.n_policy_noise_warmup_steps
                        policy_noise = ratio * self.args.policy_noise + (1 - ratio) * self.args.init_policy_noise
                    else:
                        policy_noise = None
                    with torch.no_grad():
                        action_left, _, _ = self.actor_left.get_action(obs_left_tensor, policy_noise)
                        action_right, _, _ = self.actor_right.get_action(obs_right_tensor, policy_noise)
                    
                    action_left = action_left.cpu().numpy().flatten()
                    action_right = action_right.cpu().numpy().flatten()
    
            self.stored_action_left = action_left
            self.stored_action_right = action_right
        else:
            action_left = self.stored_action_left
            action_right = self.stored_action_right

        current_action_high = self.args.action_high
        action_left = np.clip(action_left, -current_action_high, current_action_high)
        action_right = np.clip(action_right, -current_action_high, current_action_high)
        target_hand_r[self.args.action_idxs] += action_right
        target_hand_l[self.args.action_idxs] += action_left
        target_hand_r = np.clip(target_hand_r, -0.3, RIGHT_JOINT_LIMITS[:, 1])
        target_hand_l = np.clip(target_hand_l, -0.3, LEFT_JOINT_LIMITS[:, 1])

        # 3. Check safety of Actions
        safe_0 = time.time()
        target_hand_r, target_hand_l = self.collision_checker.get_safe_move(
            target_hand_r, target_hand_l,
            curr_r, curr_l
        )
        target_hand_r = np.clip(target_hand_r, -0.3, RIGHT_JOINT_LIMITS[:, 1])
        target_hand_l = np.clip(target_hand_l, -0.3, LEFT_JOINT_LIMITS[:, 1])
        safe_1 = time.time()
        self.env.key_on = self.get_key_on(self.global_idx)
        
        # 4. Move robot
        (next_obs_left, next_obs_right), (rew_left, rew_right), done, info = self.env.step(
            right_arm=(ee_pos_r, ee_euler_r),
            left_arm=(ee_pos_l, ee_euler_l),
            right_hand=target_hand_r,
            left_hand=target_hand_l,
        )
        if done:
            prefix = "eval_episode" if self.policy_stage == policy_stages.EVAL else "episode"
            
            metrics = self.env.episode_metrics
            for k, v in metrics.items():
                self.logger.add_scalar(f"{prefix}/{k}", v, self.global_idx)
            self.logger.add_scalar(f"{prefix}/ckpt", self.curr_ckpt, self.global_idx)

            self.stage = stages.RESETTING
            self.global_ep += 1
            self.chunk_step = 0
            self.accum_rew_left = np.zeros(3)
            self.accum_rew_right = np.zeros(3)

        self.right_history.append(next_obs_right)
        self.left_history.append(next_obs_left)
        self.episode_residual_actions.append((action_left, action_right))
        
        if done:
            self.last_episode_residual_actions = list(self.episode_residual_actions)
            self.last_episode_pressed_keys = list(self.env.music_history['pressed'])
            self.episode_residual_actions = []

        self.accum_rew_left += np.array(rew_left)
        self.accum_rew_right += np.array(rew_right)
        self.chunk_step += 1
        next_obs_left_stacked = np.concatenate(list(self.left_history))
        next_obs_right_stacked = np.concatenate(list(self.right_history))
        
        if self.chunk_step >= self.args.chunk_size or done:
            transition = dict(
                    obs_left=self.chunk_start_obs_left,
                    obs_right=self.chunk_start_obs_right,
                    action_left=action_left,
                    action_right=action_right,
                    next_obs_left=next_obs_left_stacked,
                    next_obs_right=next_obs_right_stacked,
                    rew_left=self.accum_rew_left / self.args.chunk_size,
                    rew_right=self.accum_rew_right / self.args.chunk_size,
                    masks=1.0 - float(done),
                    dones=done,
                )
            
            # Reset accumulation
            self.chunk_step = 0
            self.accum_rew_left = np.zeros(3)
            self.accum_rew_right = np.zeros(3)

            if self.policy_stage != policy_stages.EVAL:
                self.batch.append(transition)
                if len(self.batch) >= self.batch_size:
                    avg_reward_left = np.mean([x['rew_left'][0] * 0.5 + x['rew_left'][1] * 0.5 + x['rew_left'][2] for x in self.batch])
                    avg_reward_right = np.mean([x['rew_right'][0] * 0.5 + x['rew_right'][1] * 0.5 + x['rew_right'][2] for x in self.batch])
                    print(f"sending batch to learner with avg left {avg_reward_left:.4f} right {avg_reward_right:.4f}")
                    self.sock.send_pyobj((self.batch, avg_reward_left, avg_reward_right))
                    self.batch = []
        
        self.global_idx += 1
        
        loop_time = time.time() - s0

        if self.global_idx % self.args.log_freq:
            self.logger.add_scalar(f"time/policy_loop", loop_time, self.global_idx)
            self.logger.add_scalar(f"time/collision_check", safe_1 - safe_0, self.global_idx)

    def get_key_on(self, global_idx):
        for key_on, n_steps in self.args.key_on_curriculum:
            if n_steps == -1 or global_idx < n_steps:
                return key_on
        return self.args.key_on_curriculum[-1][0]

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

    def get_guided_action(self, step_idx):
        if not self.last_episode_residual_actions or not self.last_episode_pressed_keys:
            return np.zeros(self.args.action_dim // 2), np.zeros(self.args.action_dim // 2)

        # Uses window [step_idx + 0, step_idx + 6] like bimanual_offsets
        check_start = step_idx
        check_end = min(len(self.last_episode_pressed_keys), step_idx + 6)

        # Base action for this step from last episode
        idx_for_base = min(step_idx, len(self.last_episode_residual_actions) - 1)
        last_action_left = self.last_episode_residual_actions[idx_for_base][0]
        last_action_right = self.last_episode_residual_actions[idx_for_base][1]
        
        # Define mappings
        rh_fingers = [(1, 0), (2, 1), (3, 2)] 
        lh_fingers = [(6, 0), (7, 1), (8, 2)]

        def calculate_hand_correction(fingers, last_action, is_right_hand=True):
            new_action = last_action.copy()
            for f_id, a_idx in fingers:
                finger_deltas = []
                # First joint index for this finger (2 joints per finger)
                joint_start = a_idx * 2
                
                for t in range(check_start, check_end):
                    target_notes = self.env._notes[t] if t < len(self.env._notes) else []
                    pressed_mask = self.last_episode_pressed_keys[t]
                    pressed_indices = np.where(pressed_mask > 0)[0]
                    
                    target_note = next((n for n in target_notes if n.fingering == f_id), None)
                    if target_note is None: continue
                    
                    target_key = target_note.key
                    pressed_indices_for_f = self.get_pressed_indices_for_finger(pressed_indices, target_notes, f_id)
                    
                    if len(pressed_indices_for_f) == 0: continue
                    
                    dists = np.abs(pressed_indices_for_f - target_key)
                    if np.min(dists) > 3: continue
                    
                    closest_key = pressed_indices_for_f[np.argmin(dists)]
                    
                    step_delta = 0
                    if closest_key < target_key:
                        step_delta = 0.3
                    elif closest_key > target_key:
                        step_delta = -0.3
                    else:
                        if target_key + 1 in pressed_indices_for_f or target_key + 2 in pressed_indices_for_f:
                            step_delta = -0.15
                        elif target_key - 1 in pressed_indices_for_f or target_key - 2 in pressed_indices_for_f:
                            step_delta = 0.15
                    
                    if step_delta != 0:
                        finger_deltas.append(step_delta)
                
                if finger_deltas:
                    delta = np.mean(finger_deltas)
                    new_action[joint_start] += delta
                    # Spread Logic (similar to optimize_open_loop)
                    if delta > 0:
                        if a_idx < 2: new_action[(a_idx + 1) * 2] += delta * 0.3
                    elif delta < 0:
                        if a_idx > 0: new_action[(a_idx - 1) * 2] += delta * 0.3
            
            return new_action

        guided_left = calculate_hand_correction(lh_fingers, last_action_left, is_right_hand=False)
        guided_right = calculate_hand_correction(rh_fingers, last_action_right, is_right_hand=True)

        return guided_left, guided_right

    def get_guidance_sign(self, step_idx):
        if not self.last_episode_residual_actions or not self.last_episode_pressed_keys:
            return torch.zeros(self.args.action_dim // 2), torch.zeros(self.args.action_dim // 2)

        # Uses window [step_idx + 0, step_idx + 6] like bimanual_offsets
        check_start = step_idx
        check_end = min(len(self.last_episode_pressed_keys), step_idx + 6)
        
        # Define mappings
        rh_fingers = [(1, 0), (2, 1), (3, 2)] 
        lh_fingers = [(6, 0), (7, 1), (8, 2)]

        def calculate_hand_sign(fingers, is_right_hand=True):
            guidance = torch.zeros(self.args.action_dim // 2)
            for f_id, a_idx in fingers:
                finger_deltas = []
                start = a_idx * 2
                
                for t in range(check_start, check_end):
                    target_notes = self.env._notes[t] if t < len(self.env._notes) else []
                    pressed_mask = self.last_episode_pressed_keys[t]
                    pressed_indices = np.where(pressed_mask > 0)[0]
                    
                    target_note = next((n for n in target_notes if n.fingering == f_id), None)
                    if target_note is None: continue
                    
                    target_key = target_note.key
                    pressed_indices_for_f = self.get_pressed_indices_for_finger(pressed_indices, target_notes, f_id)
                    
                    if len(pressed_indices_for_f) == 0: continue
                    
                    dists = np.abs(pressed_indices_for_f - target_key)
                    if np.min(dists) > 6: continue
                    
                    closest_key = pressed_indices_for_f[np.argmin(dists)]
                    
                    step_delta = 0
                    if closest_key < target_key:
                        step_delta = 1
                    elif closest_key > target_key:
                        step_delta = -1
                    else:
                        if target_key + 1 in pressed_indices_for_f or target_key + 2 in pressed_indices_for_f:
                            step_delta = -1
                        elif target_key - 1 in pressed_indices_for_f or target_key - 2 in pressed_indices_for_f:
                            step_delta = 1
                    
                    if step_delta != 0:
                        finger_deltas.append(step_delta)

                if finger_deltas:
                    avg_delta = np.mean(finger_deltas)
                    if avg_delta > 0:
                        guidance[start] = 1
                    elif avg_delta < 0:
                        guidance[start] = -1
            
            return guidance

        guided_left = calculate_hand_sign(lh_fingers, is_right_hand=False)
        guided_right = calculate_hand_sign(rh_fingers, is_right_hand=True)

        return guided_left, guided_right



def main(args=None):
    rclpy.init(args=args)
    node = HandelbotResRLActor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        traceback.print_exc()
    finally:
        node.shutdown()
        node.restore_terminal()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

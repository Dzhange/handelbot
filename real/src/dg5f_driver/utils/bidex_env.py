
import os
import pickle
import time
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import mido
import numpy as np
import torch
from control_msgs.msg import MultiDOFCommand
from rclpy.node import Node
from sensor_msgs.msg import JointState
from sklearn.metrics import precision_recall_fscore_support

from . import common_utils
from .arm_controller import ArmController
from .bidex_env_config import BiDexEnvConfig
from .plot_utils import ArmTracker, FingerTracker
from .robot_utils import get_ori, get_waypoint

class BiDexEnv:
    def __init__(self, cfg: BiDexEnvConfig, node: Node):
        self.cfg = cfg
        self.node = node

        # Initialize Arms
        self.controller_right = None
        self.controller_left = None
        self.home_pos_right = None
        self.home_euler_right = None
        self.home_pos_left = None
        self.home_euler_left = None

        if self.cfg.enable_right_arm:
            self.controller_right = ArmController(self.cfg.right_arm_controller)
            self.controller_right.reset(randomize=False)
            proprio = self.controller_right.get_proprio()
            self.home_pos_right = proprio.eef_pos
            self.home_euler_right = proprio.eef_euler
            print("Right Arm controller initialized")

        if self.cfg.enable_left_arm:
            self.controller_left = ArmController(self.cfg.left_arm_controller)
            self.controller_left.reset(randomize=False)
            proprio = self.controller_left.get_proprio()
            self.home_pos_left = proprio.eef_pos
            self.home_euler_left = proprio.eef_euler
            print("Left Arm controller initialized")

        # Initialize Hands
        self.right_hand_pub = None
        self.left_hand_pub = None
        
        self.latest_right_hand_qpos = np.zeros(20) 
        self.latest_left_hand_qpos = np.zeros(20)

        if self.cfg.enable_right_hand:
            self.right_hand_pub = self.node.create_publisher(
                MultiDOFCommand, '/dg5f_right/policy_reference', 30
            )
            self.node.create_subscription(
                JointState, '/dg5f_right/joint_states', self.right_hand_state_callback, 10
            )
            self.right_hand_joint_names = [f"rj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]

        if self.cfg.enable_left_hand:
            self.left_hand_pub = self.node.create_publisher(
                MultiDOFCommand, '/dg5f_left/policy_reference', 30
            )
            self.node.create_subscription(
                JointState, '/dg5f_left/joint_states', self.left_hand_state_callback, 10
            )
            self.left_hand_joint_names = [f"lj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5)]

        # Trackers
        self.right_finger_tracker = None
        self.left_finger_tracker = None
        self.right_arm_tracker = None
        self.left_arm_tracker = None
        
        right_arm_joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        left_arm_joint_names = [f"panda_left_joint{i}" for i in range(1, 8)]

        self.start_time = time.time()
        
        if self.cfg.enable_finger_plot:
            if self.cfg.enable_right_hand:
                self.right_finger_tracker = FingerTracker(self.right_hand_joint_names)
            if self.cfg.enable_left_hand:
                self.left_finger_tracker = FingerTracker(self.left_hand_joint_names)
        
        if self.cfg.enable_arm_plot:
            if self.cfg.enable_right_arm:
                self.right_arm_tracker = ArmTracker(right_arm_joint_names)
            if self.cfg.enable_left_arm:
                self.left_arm_tracker = ArmTracker(left_arm_joint_names)

        self.joint_history = deque(maxlen=1000)
        self.action_history = deque(maxlen=1000)
        self.prev_time = time.time()
        
        self.prev_left_hand_action = None
        self.prev_right_hand_action = None

    def _now_s(self):
        return time.time() - self.start_time

    def right_hand_state_callback(self, msg: JointState):
        positions = []
        for joint_name in self.right_hand_joint_names:
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                positions.append(msg.position[idx])
        if len(positions) == 20:
            self.latest_right_hand_qpos = np.array(positions)

    def left_hand_state_callback(self, msg: JointState):
        positions = []
        for joint_name in self.left_hand_joint_names:
            if joint_name in msg.name:
                idx = msg.name.index(joint_name)
                positions.append(msg.position[idx])
        if len(positions) == 20:
            self.latest_left_hand_qpos = np.array(positions)

    def publish_right_hand(self, qpos):
        if self.right_hand_pub:
            msg = MultiDOFCommand()
            msg.dof_names = self.right_hand_joint_names
            qpos[:4] = 0 # set thumb to 0
            msg.values = qpos.tolist()
            self.right_hand_pub.publish(msg)

    def publish_left_hand(self, qpos):
        if self.left_hand_pub:
            msg = MultiDOFCommand()
            msg.dof_names = self.left_hand_joint_names
            qpos[:4] = 0 # set thumb to 0
            msg.values = qpos.tolist()
            self.left_hand_pub.publish(msg)

    def get_right_hand_proprio(self):
        return self.latest_right_hand_qpos

    def get_left_hand_proprio(self):
        return self.latest_left_hand_qpos

    def get_right_arm_proprio(self):
        if self.controller_right:
            return self.controller_right.get_proprio()
        return None

    def get_left_arm_proprio(self):
        if self.controller_left:
            return self.controller_left.get_proprio()
        return None
    
    def _move_to_controller(self, controller, target_pos, target_euler, control_freq):
        if controller is None: return
        proprio = controller.get_proprio()
        
        gen_waypoint, num_steps = get_waypoint(proprio.eef_pos, target_pos, max_delta=0.01)
        gen_ori = get_ori(proprio.eef_euler, target_euler, num_steps)

        for i in range(1, num_steps + 1):
            next_ee_pos = gen_waypoint(i)
            next_ee_euler = gen_ori(i)

            with common_utils.FreqGuard(control_freq):
                controller.position_control(next_ee_pos, next_ee_euler)

    def right_move_to(self, pos, euler, control_freq=10):
        self._move_to_controller(self.controller_right, pos, euler, control_freq)

    def left_move_to(self, pos, euler, control_freq=10):
        self._move_to_controller(self.controller_left, pos, euler, control_freq)
            
    def go_home(self):
        if self.controller_right:
            self.controller_right.reset(randomize=False)
        if self.controller_left:
            self.controller_left.reset(randomize=False)
        zeros = np.zeros(20)
        self.publish_right_hand(zeros)
        self.publish_left_hand(zeros)
        
    def reset(self):
        obs = {
            'right_arm': self.get_right_arm_proprio(),
            'right_hand': self.get_right_hand_proprio(),
            'left_arm': self.get_left_arm_proprio(),
            'left_hand': self.get_left_hand_proprio()
        }
        self.prev_left_hand_action = None
        self.prev_right_hand_action = None
        return obs

    def step(self, left_arm=None, left_hand=None, right_arm=None, right_hand=None):
        current_s = self._now_s()

        if self.controller_right and right_arm is not None:
            pos, euler = right_arm
            self.controller_right.position_control(pos, euler)

        if self.controller_left and left_arm is not None:
            pos, euler = left_arm
            self.controller_left.position_control(pos, euler)

        if right_hand is not None:
            self.publish_right_hand(right_hand)
            if self.right_finger_tracker:
                self.right_finger_tracker.add_target(right_hand, current_s)

        if left_hand is not None:
            self.publish_left_hand(left_hand)
            if self.left_finger_tracker:
                self.left_finger_tracker.add_target(left_hand, current_s)

        right_arm_proprio = self.get_right_arm_proprio()
        left_arm_proprio = self.get_left_arm_proprio()
        right_hand_proprio = self.get_right_hand_proprio()
        left_hand_proprio = self.get_left_hand_proprio()

        if right_hand is not None and self.right_finger_tracker:
            self.right_finger_tracker.add_actual(right_hand_proprio, np.zeros(20), current_s)

        if left_hand is not None and self.left_finger_tracker:
            self.left_finger_tracker.add_actual(left_hand_proprio, np.zeros(20), current_s)

        obs = {
            'right_arm': right_arm_proprio,
            'right_hand': right_hand_proprio,
            'left_arm': left_arm_proprio,
            'left_hand': left_hand_proprio
        }

        # Track history
        curr_joints = []
        if right_arm_proprio:
             curr_joints.append(right_arm_proprio.joint_pos)
        else:
             curr_joints.append(np.zeros(7))
        curr_joints.append(right_hand_proprio)
        if left_arm_proprio:
             curr_joints.append(left_arm_proprio.joint_pos)
        else:
             curr_joints.append(np.zeros(7))
        curr_joints.append(left_hand_proprio)
        self.joint_history.append(np.concatenate(curr_joints))

        # Track Actions
        curr_actions = []
        if right_arm is not None:
             r_pos, r_euler = right_arm
             curr_actions.append(np.concatenate([r_pos, r_euler]))
        else:
             curr_actions.append(np.zeros(6))
        if right_hand is not None:
             curr_actions.append(right_hand)
        else:
             curr_actions.append(np.zeros(20))
        if left_arm is not None:
             l_pos, l_euler = left_arm
             curr_actions.append(np.concatenate([l_pos, l_euler]))
        else:
             curr_actions.append(np.zeros(6))
        if left_hand is not None:
             curr_actions.append(left_hand)
        else:
             curr_actions.append(np.zeros(20))
        self.action_history.append(np.concatenate(curr_actions))
        
        return obs, 0, False, {}

    def plot(self, output_dir=None):
        if output_dir:
            print("plotting to", output_dir)
            if self.right_finger_tracker:
                self.right_finger_tracker.plot(os.path.join(output_dir, "right_finger.jpg"))
            if self.left_finger_tracker:
                self.left_finger_tracker.plot(os.path.join(output_dir, "left_finger.jpg"))
            if self.right_arm_tracker:
                self.right_arm_tracker.plot(os.path.join(output_dir, "right_arm.jpg"))
            if self.left_arm_tracker:
                self.left_arm_tracker.plot(os.path.join(output_dir, "left_arm.jpg"))
            
            try:
                np.save(os.path.join(output_dir, "last_1000_joint_pos.npy"), np.array(self.joint_history))
                np.save(os.path.join(output_dir, "last_1000_actions.npy"), np.array(self.action_history))
            except Exception as e:
                print(f"Failed to save history: {e}")


class MidiKeyTracker:
    def __init__(self, port_name='MIDI KEYBOARD:MIDI KEYBOARD MIDI 1 24:0'):
        self.port_name = os.environ.get('MIDI_PORT', port_name)
        self.keys = np.zeros(88, dtype=int)
        try:
            self.inport = mido.open_input(self.port_name)
            print(f"Opened MIDI port: {self.port_name}")
        except Exception as e:
            print(f"FAILED to open MIDI port: {e}")
            self.inport = None

    def update(self):
        if self.inport:
            for msg in self.inport.iter_pending():
                if msg.type == 'note_on' and msg.velocity > 0:
                    if 21 <= msg.note <= 108:
                        self.keys[msg.note - 21] = 1
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if 21 <= msg.note <= 108:
                        self.keys[msg.note - 21] = 0
        return self.keys

    def close(self):
        if self.inport:
            self.inport.close()


class RealPianoSeparateWrapper:
    def __init__(self, env: BiDexEnv, sheet_music_path: str, device: str = 'cuda', key_on: float = 0.7, coef_action_l1: float = 0.0):
        self.env = env
        self.device = device
        self.key_on = key_on
        self.coef_action_l1 = coef_action_l1
        self.midi_tracker = MidiKeyTracker()
        
        self.n_keys = 88
        self.n_steps_lookahead = 5
        self.n_fingering_lookahead = 5
        
        song_scale = max(1, 10 // self.env.cfg.policy_freq)
        if "hotcross" in sheet_music_path:
            song_scale = 2
        self.load_song(sheet_music_path, song_scale)
        
        self.song_t_idx = 0
        self.pressed_history = []
        self.goal_history = []
        self.episode_reward = 0
        self.split_idx = 45 # 0-44 (45 keys) for Left, 45-87 (43 keys) for Right

        self.best_f1 = -1.0
        self.episode_traj_history = []
        self.log_dir = None
        self.is_eval = False
        self.global_idx = 0
        self.target_qpos_hand_left = None
        self.target_qpos_hand_right = None

    def get_log_info(self, info):
        log_dict = {}
        if 'midi_keys' in info:
             pressed = info['midi_keys'] 
             goal_state = self.get_goal_state(self.song_t_idx)
             goal_current = goal_state[0]
             
             def _calc_stats(p, g):
                 tp = np.sum(p * g)
                 fp = np.sum(p * (1 - g))
                 fn = np.sum((1 - p) * g)
                 precision = tp / (tp + fp + 1e-8)
                 recall = tp / (tp + fn + 1e-8)
                 f1 = 2 * precision * recall / (precision + recall + 1e-8)
                 return precision, recall, f1

             log_dict['precision'], log_dict['recall'], log_dict['f1'] = _calc_stats(pressed, goal_current)
             log_dict['lh_precision'], log_dict['lh_recall'], log_dict['lh_f1'] = _calc_stats(pressed[:self.split_idx], goal_current[:self.split_idx])
             log_dict['rh_precision'], log_dict['rh_recall'], log_dict['rh_f1'] = _calc_stats(pressed[self.split_idx:], goal_current[self.split_idx:])

             log_dict['num_keys_pressed'] = np.sum(pressed)
             log_dict['num_keys_requested'] = np.sum(goal_current)
        return log_dict

    def get_eval_episode_metrics(self):
        if len(self.pressed_history) == 0:
            return {}
            
        y_preds = np.array(self.pressed_history)
        y_trues = np.array(self.goal_history)
        
        y_preds_flat = y_preds.flatten()
        y_trues_flat = y_trues.flatten()
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=y_trues_flat, y_pred=y_preds_flat, average="binary", zero_division=1
        )
        
        y_preds_lh = y_preds[:, :self.split_idx].flatten()
        y_trues_lh = y_trues[:, :self.split_idx].flatten()
        lh_precision, lh_recall, lh_f1, _ = precision_recall_fscore_support(
            y_true=y_trues_lh, y_pred=y_preds_lh, average="binary", zero_division=1
        )

        y_preds_rh = y_preds[:, self.split_idx:].flatten()
        y_trues_rh = y_trues[:, self.split_idx:].flatten()
        rh_precision, rh_recall, rh_f1, _ = precision_recall_fscore_support(
            y_true=y_trues_rh, y_pred=y_preds_rh, average="binary", zero_division=1
        )
        
        return dict(
            precision=precision, recall=recall, f1=f1,
            lh_f1=lh_f1, rh_f1=rh_f1,
            lh_precision=lh_precision, lh_recall=lh_recall,
            rh_precision=rh_precision, rh_recall=rh_recall,
            len=len(self.pressed_history),
            episode_reward=self.episode_reward
        )

    def plot_notes(self, save_path):
        if len(self.music_history['pressed']) == 0:
            return

        y_preds = np.array(self.music_history['pressed']) 
        y_trues = np.array(self.music_history['goal']) 
        
        fig, ax = plt.subplots(figsize=(15, 8))
        tp = (y_preds == 1) & (y_trues == 1)
        fp = (y_preds == 1) & (y_trues == 0)
        fn = (y_preds == 0) & (y_trues == 1)
        
        ax.scatter(*np.where(tp)[::-1], c='green', s=40, label='Correct (TP)', alpha=0.6)
        ax.scatter(*np.where(fp)[::-1], c='red', s=10, label='Wrong (FP)', alpha=0.6)
        ax.scatter(*np.where(fn)[::-1], c='orange', s=40, label='Missed (FN)', alpha=0.6)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Midi Key Index (0-87)')
        ax.set_title('Piano Roll Analysis')
        ax.legend()
        ax.axhline(y=self.split_idx, color='blue', linestyle='--', label='Hand Split')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

    def load_song(self, path, control_scale):
        with open(path, 'rb') as f:
            note_traj = pickle.load(f)
        note_traj = note_traj.trim_silence()
        note_traj.add_initial_buffer_time(0.5)
        self._notes = note_traj.notes[::control_scale]
        if "bach" in path:
            self._notes = [n for n in self._notes for _ in range(2)]
        print(f"Loaded song from {path} with {len(self._notes)} steps")

    def get_goal_state(self, t_idx):
        goal_state = np.zeros((self.n_steps_lookahead + 1, self.n_keys))
        t_start = t_idx
        t_end = min(t_start + (self.n_steps_lookahead + 1), len(self._notes))
        for i, t in enumerate(range(t_start, t_end)):
            keys = [note.key for note in self._notes[t]]
            if keys:
                 goal_state[i, keys] = 1.0
        return goal_state

    def get_fingering_state(self, t_idx):
        fingering_state = np.zeros((self.n_fingering_lookahead + 1, 2, 5))
        for t_offset in range(self.n_fingering_lookahead + 1):
            t_curr = min(t_idx + t_offset, len(self._notes) - 1)
            if t_curr < 0: continue
            notes = self._notes[t_curr] 
            for note in notes:
                if note.fingering < 5:
                     fingering_state[t_offset, 1, note.fingering] = 1.0
                else:
                     fingering_state[t_offset, 0, note.fingering - 5] = 1.0
        return fingering_state

    def _get_obs(self, env_obs):
        midi_keys = self.midi_tracker.update().copy()
        goal_state = self.get_goal_state(self.song_t_idx)
        fingering_state = self.get_fingering_state(self.song_t_idx)
        
        midi_left = midi_keys[:self.split_idx]
        goal_left = goal_state[:, :self.split_idx].flatten()
        fingering_left = fingering_state[:, 1, :].flatten()
        
        midi_right = midi_keys[self.split_idx:]
        goal_right = goal_state[:, self.split_idx:].flatten()
        fingering_right = fingering_state[:, 0, :].flatten()

        # Proprio Left
        proprio_left_list = []
        if env_obs['left_arm'] is not None:
             proprio_left_list.append(env_obs['left_arm'].eef_pos_euler)
        if env_obs['left_hand'] is not None:
             h = env_obs['left_hand']
             proprio_left_list.append(h)
        proprio_left = np.concatenate(proprio_left_list)
        obs_left = np.concatenate([proprio_left, goal_left, fingering_left, midi_left])

        # Proprio Right
        proprio_right_list = []
        if env_obs['right_arm'] is not None:
             proprio_right_list.append(env_obs['right_arm'].eef_pos_euler)
        if env_obs['right_hand'] is not None:
             h = env_obs['right_hand']
             proprio_right_list.append(h)
        proprio_right = np.concatenate(proprio_right_list)
        obs_right = np.concatenate([proprio_right, goal_right, fingering_right, midi_right])

        return (obs_left, obs_right), {'midi_keys': midi_keys}

    def get_reward(self, info, action_l=None, action_r=None):
        pressed = info['midi_keys']
        goal = self.get_goal_state(self.song_t_idx)[0]

        def calc_hand_rew(p, g):
            on = np.nonzero(g)[0]
            off = np.nonzero(1 - g)[0]
            rew_on = p[on].mean() if len(on) > 0 else 0
            rew_off = float(1 - p[off].any())
            return rew_on, rew_off

        rew_l_on, rew_l_off = calc_hand_rew(pressed[:self.split_idx], goal[:self.split_idx])
        rew_r_on, rew_r_off = calc_hand_rew(pressed[self.split_idx:], goal[self.split_idx:])
        
        rl = self.key_on * rew_l_on + (1 - self.key_on) * rew_l_off
        rr = self.key_on * rew_r_on + (1 - self.key_on) * rew_r_off
        
        if self.coef_action_l1 != 0:
            if action_l is not None: rl += self.coef_action_l1 * np.linalg.norm(action_l, ord=1)
            if action_r is not None: rr += self.coef_action_l1 * np.linalg.norm(action_r, ord=1)
            
        return (rl, rr), {}

    def step(self, **kwargs):
        env_obs, _, _, _ = self.env.step(**kwargs)
        self.episode_traj_history.append((kwargs.get('left_arm'), kwargs.get('left_hand'), kwargs.get('right_arm'), kwargs.get('right_hand')))

        obs, info = self._get_obs(env_obs)
        (rew_l, rew_r), _ = self.get_reward(info, action_l=kwargs.get('residual_action_left'), action_r=kwargs.get('residual_action_right'))
        
        info['log'] = self.get_log_info(info)
        self.pressed_history.append(info['midi_keys'])
        self.goal_history.append(self.get_goal_state(self.song_t_idx)[0])

        self.episode_reward += (rew_l + rew_r) 
        self.song_t_idx += 1
        
        done = False
        if self.song_t_idx >= self.data_len:
            done = True
            self.music_history = dict(pressed=self.pressed_history.copy(), goal=self.goal_history.copy())
            self.episode_metrics = self.get_eval_episode_metrics()
            print("Episode Metrics:", self.episode_metrics)

            if self.log_dir:
                if self.is_eval:
                    eval_dir = os.path.join(self.log_dir, "eval_data")
                    os.makedirs(eval_dir, exist_ok=True)
                    self.plot_notes(os.path.join(eval_dir, f"eval_traj_{self.global_idx}_plot.jpg"))

                if self.episode_metrics.get('f1', -1.0) > self.best_f1:
                    self.best_f1 = self.episode_metrics['f1']
                    best_dir = os.path.join(self.log_dir, "best_traj")
                    os.makedirs(best_dir, exist_ok=True)
                    self.plot_notes(os.path.join(best_dir, "best_plot.jpg"))

            self.song_t_idx = 0
            self.pressed_history = []
            self.goal_history = []
            self.episode_traj_history = []

        return obs, (rew_l, rew_r), done, info

    def reset(self):
        self.pressed_history = []
        self.goal_history = []
        self.episode_traj_history = []
        env_obs = self.env.reset()
        self.song_t_idx = 0
        self.midi_tracker.update() 
        self.episode_reward = 0
        return self._get_obs(env_obs)[0]

    def shutdown(self):
        self.midi_tracker.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


class Sim2RealWrapper(RealPianoSeparateWrapper):
    def __init__(self, env: BiDexEnv, sheet_music_path: str, device: str = 'cuda', key_on: float = 0.5, 
                 use_sim_obs: bool = False, sim_kwargs: dict = None, coef_action_l1: float = 0.0,
                 visualize_maniskill: bool = False):
        super().__init__(env, sheet_music_path, device, key_on, coef_action_l1)
        self.use_sim_obs = use_sim_obs
        self.visualize_maniskill = visualize_maniskill
        
        # Override lookahead for Sim2Real policies which often use more
        self.n_steps_lookahead = 10 
        self.n_fingering_lookahead = 10
        song_scale = max(1, 10 // self.env.cfg.policy_freq)
        if "hotcross" in sheet_music_path:
            song_scale = 2
        self.load_song(sheet_music_path, song_scale)

        self.hand_indices = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18])
        remap = [0, 5, 10, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14]
        self.hand_indices_remap = self.hand_indices[remap]
        self.fixed_indices = [7, 11, 15, 19]
        self.obs_offsets = None

        if self.use_sim_obs:
            print("Initializing simulation environment...")
            default_sim_kwargs = dict(
                env_id="Piano-Bimanual-v1", obs_mode="state", control_mode="pd_ee_script_xy_pos_joint_delta_pos",
                render_mode="human" if self.visualize_maniskill else "rgb_array", sim_backend="auto", num_envs=1, is_eval=True,
                sim_config=dict(control_freq=10), robot_uids=("delto_left_panda_reduced_fixed1", "delto_right_panda_reduced_fixed1")
            )
            if sim_kwargs: default_sim_kwargs.update(sim_kwargs)
            eid = default_sim_kwargs.pop("env_id")
            default_sim_kwargs.update({"key_on": key_on, "note_trajectory_name": sheet_music_path, "midi_name": None})
            
            self.sim_env = gym.make(eid, **default_sim_kwargs)
            from envs import FlattenPianoActionSpaceWrapper
            from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
            self.sim_env = FlattenPianoActionSpaceWrapper(self.sim_env, default_sim_kwargs.get("control_mode"), default_sim_kwargs.get("use_3d_finger_actions", False))
            self.sim_env = ManiSkillVectorEnv(self.sim_env, 1, ignore_terminations=True, record_metrics=True)
            self.sim_obs = self.sim_env.reset()[0].cpu().numpy()[0]
            if self.visualize_maniskill:
                self.sim_env.render()
        else:
            self.sim_env = None

    def _get_obs(self, env_obs):
        mk = self.midi_tracker.update().copy()
        gs = self.get_goal_state(self.song_t_idx)
        fs = self.get_fingering_state(self.song_t_idx)

        # Left Arm/Hand
        if self.use_sim_obs:
            l_ee = self.sim_obs[:6]
            l_hand = self.sim_obs[6:21]
        else:
            l_ee = env_obs['left_arm'].eef_pos_euler.copy()
            if self.obs_offsets:
                # starting xyz of end-effector in ManiSkill sim.
                l_ee[0] -= self.obs_offsets['left_x'] - 0.615 
                l_ee[1] -= self.obs_offsets['left_y'] - 0.25
                l_ee[2] -= self.obs_offsets['left_z'] - 0.38
            l_hand = env_obs['left_hand']
            if len(l_hand) == 20: l_hand = l_hand[self.hand_indices_remap]
        
        # Right Arm/Hand
        if self.use_sim_obs:
            r_ee = self.sim_obs[21:27]
            r_hand = self.sim_obs[27:42]
        else:
            r_ee = env_obs['right_arm'].eef_pos_euler.copy()
            if self.obs_offsets:
                # starting xyz of end-effector in ManiSkill sim.
                r_ee[0] -= self.obs_offsets['right_x'] - 0.615
                r_ee[1] -= self.obs_offsets['right_y'] + 0.25
                r_ee[2] -= self.obs_offsets['right_z'] - 0.38
            r_hand = env_obs['right_hand']
            if len(r_hand) == 20: r_hand = r_hand[self.hand_indices_remap]

        lookahead89 = np.zeros((11, 89))
        lookahead89[:, :88] = gs
        
        full_obs = np.concatenate([l_ee, l_hand, r_ee, r_hand, mk, lookahead89.flatten(), fs.flatten()])
        return full_obs, {'midi_keys': mk}

    def step(self, sim_action=None, **kwargs):
        if self.use_sim_obs and self.sim_env and sim_action is not None:
            self.sim_obs = self.sim_env.step(torch.tensor(sim_action).unsqueeze(0))[0].cpu().numpy()[0]
            if self.visualize_maniskill: self.sim_env.render()

        env_obs, _, _, _ = self.env.step(**kwargs)
        obs, info = self._get_obs(env_obs)
        
        def calc_rew(p, g):
            on = np.nonzero(g)[0]
            off = np.nonzero(1-g)[0]
            rew_on = p[on].mean() if len(on) > 0 else 0
            rew_off = float(1 - p[off].any())
            return self.key_on * rew_on + (1 - self.key_on) * rew_off
            
        rew = calc_rew(info['midi_keys'], self.get_goal_state(self.song_t_idx)[0])
        if self.coef_action_l1 and sim_action is not None:
             rew += self.coef_action_l1 * np.linalg.norm(sim_action, 1)

        info['log'] = self.get_log_info(info)
        self.pressed_history.append(info['midi_keys'])
        self.goal_history.append(self.get_goal_state(self.song_t_idx)[0])
        self.episode_reward += rew
        self.song_t_idx += 1
        
        done = False
        if self.song_t_idx >= self.data_len:
            done = True
            self.music_history = dict(pressed=self.pressed_history.copy(), goal=self.goal_history.copy())
            m = self.get_eval_episode_metrics()
            print("Episode Metrics (Sim2Real):", m)
            if self.log_dir:
                if self.is_eval:
                    self.plot_notes(os.path.join(self.log_dir, f"eval_{self.global_idx}.jpg"))
                if m.get('f1', 0) > self.best_f1:
                    self.best_f1 = m['f1']
                    self.plot_notes(os.path.join(self.log_dir, "best_plot.jpg"))
            self.song_t_idx = 0
            self.pressed_history = []
            self.goal_history = []
            if self.use_sim_obs:
                self.sim_obs = self.sim_env.reset()[0].cpu().numpy()[0]

        return obs, rew, done, info

    def reset(self):
        self.pressed_history = []
        self.goal_history = []
        self.song_t_idx = 0
        self.episode_reward = 0
        if self.use_sim_obs:
            self.sim_obs = self.sim_env.reset()[0].cpu().numpy()[0]
        return self._get_obs(self.env.reset())[0]

    def clip_and_scale_action(self, action, low, high):
        action = np.clip(action, -1, 1) if isinstance(action, np.ndarray) else torch.clip(action, -1, 1)
        return 0.5 * (high + low) + 0.5 * (high - low) * action

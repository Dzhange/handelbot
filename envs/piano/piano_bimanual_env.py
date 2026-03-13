import os
import pickle
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import sapien
import torch
from sklearn.metrics import precision_recall_fscore_support
from transforms3d.euler import euler2quat, quat2euler

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

from . import music
from .music import constants as consts
from .music import midi_file
from .rewards import CompositeReward, FingeringReward, KeyPressReward


def quat_to_euler(q):
    w, x, y, z = q.unbind(-1)

    t0 = +2.0 * (w*x + y*z)
    t1 = +1.0 - 2.0 * (x*x + y*y)
    roll = torch.atan2(t0, t1)

    t2 = +2.0 * (w*y - z*x)
    t2 = t2.clamp(-1.0, 1.0)
    pitch = torch.asin(t2)

    t3 = +2.0 * (w*z + x*y)
    t4 = +1.0 - 2.0 * (y*y + z*z)
    yaw = torch.atan2(t3, t4)

    return torch.stack([roll, pitch, yaw], dim=-1) 


@register_env("Piano-Bimanual-v1")
class PianoBimanualEnv(BaseEnv):
    def __init__(self, *args, 
                # --- Basic Setup ---
                robot_uids=("delto_left_panda_reduced_fixed1", "delto_right_panda_reduced_fixed1"), 
                horizon=150,
                piano_xyz=[0.965, 0, 0.145],
                is_eval=False,
                init_keyframe='piano_bimanual',

                # --- Music / MIDI Loading ---
                midi_name=None, 
                note_trajectory_name=None, 
                midi_start_from=0,
                trim_silence=True,
                initial_buffer_time=0.5,

                # --- Reward / RL Parameters ---
                coef_key=1.0, 
                coef_finger=1.0, 
                coef_action_l1=0.0,
                key_on=0.5,
                reward_fingering=True,
                n_steps_lookahead=10,
                n_fingering_lookahead=10,

                # --- Domain Randomization ---
                domain_rand_gains=False, 
                domain_rand_keys=False, 
                domain_rand_scene=False,
                **kwargs):
        
        # --- Environment State & Setup ---
        self.init_keyframe = init_keyframe
        self._is_eval = is_eval
        self.max_episode_steps = horizon
        self.piano_xyz = piano_xyz
        self._piano_xyz_th = None
        self._key_press_threshold = 0.04
        self.t_idx = 0 
        self.key_presses = []
        self._piano_poses = None
        self._wrist_trajectory_left = None
        self._wrist_trajectory_right = None

        # --- Music & Trajectory Setup ---
        self.midi_name = midi_name
        self.note_trajectory_name = note_trajectory_name
        self.midi_start_from = midi_start_from
        self.trim_silence = trim_silence
        self.initial_buffer_time = initial_buffer_time

        # --- Reward & RL Parameters ---
        self.coef_key = coef_key
        self.coef_finger = coef_finger
        self.coef_action_l1 = coef_action_l1
        self.key_on = key_on
        self.reward_fingering = reward_fingering
        self.n_steps_lookahead = n_steps_lookahead
        self.n_fingering_lookahead = n_fingering_lookahead

        # --- Domain Randomization ---
        self._domain_rand_gains = domain_rand_gains
        self._domain_rand_keys = domain_rand_keys
        self._domain_rand_scene = domain_rand_scene

        # setup piano
        if self.note_trajectory_name is None and self.midi_name is None:
            raise ValueError("Either `note_trajectory_name` or `midi_name` must be specified.")
        if self.note_trajectory_name is not None and self.midi_name is not None:
            raise ValueError("Only one of `note_trajectory_name` or `midi_name` can be specified.")
        
        # load music
        if self.note_trajectory_name:
            with open(self.note_trajectory_name, 'rb') as f:
                note_traj = pickle.load(f)
            self._note_traj = note_traj 
            if self.trim_silence:
                self._note_traj = self._note_traj.trim_silence()
            note_traj.add_initial_buffer_time(self.initial_buffer_time) 
        else:
            midi = music.load(self.midi_name, stretch=1, shift=0)
            self._midi = midi
            if self.trim_silence:
                self._midi = self._midi.trim_silence()

        # reward function
        self.reward_fn = self._load_reward_function()

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.9], [0, 0, 0.35])
        return [CameraConfig("base_camera", pose, 512, 512, 1, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.5, 0, 1], [0.7, 0, 0.3]) # front 
        return [CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)]

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22,
                max_rigid_patch_count=2**20
            ),
            sim_freq=200
        )

    def _load_agent(self, options: dict):
        # later overriden by loading keyframe.
        right_q = euler2quat(0, -np.pi / 2, 0) 
        super()._load_agent(options, [sapien.Pose(p=[0, 0, 0.0], q=right_q), sapien.Pose(p=[0, 0, 0.0], q=right_q)])


    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene)
        self.ground.set_collision_group_bit(group=2, bit_idx=30, bit=1)

        loader = self.scene.create_urdf_loader()
        piano_builder = loader.parse(os.path.join(os.path.dirname(__file__), "assets", "piano.urdf"))["articulation_builders"][0]
        piano_builder.initial_pose = sapien.Pose(p=self.piano_xyz, q=(0, 0, 0, 1)) 
        self.piano = piano_builder.build(name="piano")
        for joint in self.piano.get_joints():
            if joint.get_type()[0] == 'revolute_unwrapped':
                joint.set_drive_properties(stiffness=2.0, damping=0.05)
        for link in self.piano.links:
            link.set_collision_group_bit(group=2, bit_idx=31, bit=1)
            link.set_disable_gravity(True)

        # load trajectory marker
        self.lh_key_markers = []
        for idx in range(5):
            viz_builder = self.scene.create_actor_builder()
            viz_builder.add_sphere_visual(radius=0.02, material=sapien.render.RenderMaterial(
                base_color=consts.COLORS[idx],
                roughness=0.5,
                metallic=0.0,
            ))
            viz_marker = viz_builder.build(name=f"lh_key_viz_{idx}")
            self.lh_key_markers.append(viz_marker)

        self.rh_key_markers = []
        for idx in range(5):
            viz_builder = self.scene.create_actor_builder()
            viz_builder.add_sphere_visual(radius=0.02, material=sapien.render.RenderMaterial(
                base_color=consts.COLORS[idx],
                roughness=0.5,
                metallic=0.0,
            ))
            viz_marker = viz_builder.build(name=f"rh_key_viz_{idx}")
            self.rh_key_markers.append(viz_marker)

        if self._domain_rand_gains and not self._is_eval:
            self._randomize_gains()
        if self._domain_rand_keys and not self._is_eval:
            self._randomize_keys()
        if self._domain_rand_scene and not self._is_eval:
            self._randomize_scene()


    def _load_reward_function(self):
        terms = [
            KeyPressReward(coef=self.coef_key, key_on=self.key_on),
            FingeringReward(coef=self.coef_finger),
        ]
        return CompositeReward(terms)


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_vars(env_idx)
        self._initialize_agent(env_idx)
        self._initialize_actors(env_idx)
        self._initialize_piano(env_idx)

        self.eval_episode_metrics = self.get_eval_episode_metrics()
        self.key_presses = []

    
    def _initialize_vars(self, env_idx: torch.Tensor):
        self.t_idx = 0

    def _randomize_scene(self):
        noise = np.random.uniform(-0.01, 0.01, size=3)
        self.piano_xyz = [
            self.piano_xyz[0] + noise[0],
            self.piano_xyz[1] + noise[1],
            self.piano_xyz[2] + noise[2]
        ]

    def _randomize_gains(self):
        for agent in self.agent.agents:
            for joint in agent.robot.active_joints:
                for obj in joint._objs:
                    stiffness = np.random.uniform(0.9, 2) * obj.stiffness
                    damping = np.random.uniform(0.9, 2) * obj.damping
                    obj.set_drive_properties(
                        stiffness=stiffness,
                        damping=damping
                    )

    def _randomize_keys(self):
        from sapien.physx import PhysxArticulationLinkComponent, PhysxRigidBodyComponent

        for joint in self.piano.get_joints():
            if joint.get_type()[0] == 'revolute_unwrapped':
                stiffness = np.random.uniform(1, 3) * 2.0  
                damping = np.random.uniform(0.8, 1.2) * 0.05  
                joint.set_drive_properties(stiffness=stiffness, damping=damping)
        
        for link in self.piano.links:
            for obj in link._objs:
                if isinstance(obj, PhysxArticulationLinkComponent):
                    rigid_body_component = obj
                elif isinstance(obj, PhysxRigidBodyComponent):
                    rigid_body_component = obj
                else:
                    rigid_body_component = None
                if rigid_body_component is not None:
                    for shape in rigid_body_component.collision_shapes:
                        base_static_friction = 2.0
                        base_dynamic_friction = 1.0
                        base_restitution = 0.8

                        friction_scale = np.random.uniform(0.8, 1.2)
                        restitution_scale = np.random.uniform(0.8, 1.2)

                        mat = shape.physical_material
                        mat.static_friction = base_static_friction * friction_scale
                        mat.dynamic_friction = base_dynamic_friction * friction_scale
                        mat.restitution = base_restitution * restitution_scale

        self._key_press_threshold = np.random.uniform(0.04, 0.06)

    def _initialize_actors(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            agent = self.agent.agents[0] 
            stiffness = torch.tensor(agent.joint_stiffness)
            damping = torch.tensor(agent.joint_damping)
            force_limit = torch.tensor(agent.joint_force_limit)

            num_envs = agent.robot.dof.shape[0]
            dof = agent.robot.dof[0]

            self.controller_param = (
                stiffness.expand(num_envs, dof),
                damping.expand(num_envs, dof),
                force_limit.expand(num_envs, dof),
            )

    def _initialize_agent(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            for idx, agent in enumerate(self.agent.agents):
                agent.robot.set_qvel(agent.keyframes[self.init_keyframe].qvel)
                
                qpos = agent.keyframes[self.init_keyframe].qpos
                pose = agent.keyframes[self.init_keyframe].pose
                if self._domain_rand_scene and not self._is_eval:
                    pos_noise = np.random.uniform(-0.01, 0.01, size=3)
                    pose_p = pose.p + pos_noise
                    pose = sapien.Pose(p=pose_p, q=pose.q)
                    qpos_noise = np.random.uniform(-0.02, 0.02, size=qpos.shape)
                    qpos = qpos + qpos_noise

                agent.robot.set_qpos(qpos)
                agent.robot.set_pose(pose)


    def _initialize_piano(self, env_idx):
        self.piano.set_qpos(torch.zeros_like(self.piano.get_qpos()))
        if hasattr(self, "_midi"):
            note_traj = midi_file.NoteTrajectory.from_midi(
                self._midi, self.control_timestep # / 2
            )
            note_traj.add_initial_buffer_time(self.initial_buffer_time)
        else:
            note_traj = self._note_traj

        if "hotcross" in self.note_trajectory_name:
            control_scale = 2
        else:
            control_scale = 1
        self._notes = note_traj.notes[self.midi_start_from::control_scale]
        self._sustains = note_traj.sustains[self.midi_start_from::control_scale]

        if "prelude" in self.note_trajectory_name:
            factor = 2
            self._notes = [n for n in self._notes for _ in range(factor)]
            self._sustains = [s for s in self._sustains for _ in range(factor)]

    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):
        is_hybrid = False
        script_xy = False
        for v in self.control_mode.values():
            if "ee_script_xy" in v:
                is_hybrid = True
                script_xy = True
        if is_hybrid:
            action = self._hybrid_action(action, script_xy=script_xy)
        action = self._step_action(action)

        self._elapsed_steps += 1
        info = self.get_info()
        obs = self.get_obs(info)

        reward = self.get_reward(obs=obs, action=action, info=info)
        if "success" in info:
            if "fail" in info:
                terminated = torch.logical_or(info["success"], info["fail"])
            else:
                terminated = info["success"].clone()
        else:
            if "fail" in info:
                terminated = info["fail"].clone()
            else:
                terminated = torch.zeros(self.num_envs, dtype=bool, device=self.device)

        return (
            obs,
            reward,
            terminated,
            torch.zeros(self.num_envs, dtype=bool, device=self.device),
            info,
        )

    def _hybrid_action(self, raw_action, script_xy=False):
        if self.t_idx < len(self.wrist_trajectory_left):
            target_left = self.wrist_trajectory_left[self.t_idx]
            target_right = self.wrist_trajectory_right[self.t_idx]
        else:
            target_left = self.wrist_trajectory_left[-1] if len(self.wrist_trajectory_left) > 0 else np.array([0.615, 0.0])
            target_right = self.wrist_trajectory_right[-1] if len(self.wrist_trajectory_right) > 0 else np.array([0.615, 0.0])
        
        target_left_x, target_left_y = target_left
        target_right_x, target_right_y = target_right

        for k, v in raw_action.items():
            B = v.shape[0]                    

            if isinstance(v, torch.Tensor):
                xyz = torch.zeros((B,3), dtype=v.dtype, device=v.device)
                xyz[:,0] = 0.615
                xyz[:,2] = 0.38

                if "left" in k:
                    if script_xy:
                        xyz[:, 1] = target_left_y
                        xyz[:, 0] = target_left_x
                    else:
                        xyz[:, 1] = target_left_y     

                    left_q = (self.agent.agents[0].robot.pose.inv() * self.agent.agents[0].ee_pose).q
                    curr_eulers = quat_to_euler(left_q)
                    rot_x = curr_eulers[:,0]
                    rot_y = torch.full_like(rot_x, 0.0)
                    rot_z = torch.full_like(rot_x, 0)
                    rot_x = torch.where(rot_x < -2.8, -3.145,
                                    torch.where(rot_x > 2.8,  3.145, 0.0))

                    desired_rot = torch.stack([rot_x,rot_y,rot_z],dim=1)
                    raw_action[k] = torch.cat([xyz, desired_rot, v], dim=1)
                elif "right" in k:
                    if script_xy:
                        xyz[:, 1] = target_right_y
                        xyz[:, 0] = target_right_x
                    else:
                        xyz[:, 1] = target_right_y

                    right_q = (self.agent.agents[1].robot.pose.inv() * self.agent.agents[1].ee_pose).q#.cpu().numpy()
                    curr_eulers = quat_to_euler(right_q)
                    rot_x = curr_eulers[:,0]
                    rot_y = torch.full_like(rot_x,0.0)
                    rot_z = torch.full_like(rot_x, 0)
                    rot_x = torch.where(rot_x < -2.8, -3.145,
                                    torch.where(rot_x > 2.8,  3.145, 0.0))
                    desired_rot = torch.stack([rot_x,rot_y,rot_z],dim=1)

                    raw_action[k] = torch.cat([xyz, desired_rot, v], dim=1)

                else:
                    raise NotImplementedError
            else:    
                xyz = np.zeros((B,3), dtype=v.dtype)
                xyz[:,0] = 0.615
                xyz[:,2] = 0.38
                if "left" in k:
                    if script_xy:
                        xyz[:, 1] = target_left_y
                        xyz[:, 0] = target_left_x
                    else:
                        xyz[:, 1] = target_left_y         

                    left_q = (self.agent.agents[0].robot.pose.inv() * self.agent.agents[0].ee_pose).q.cpu().numpy()
                    curr_eulers = np.stack([quat2euler(q) for q in left_q], axis=0)
                    rot_x = curr_eulers[:,0]
                    rot_y = np.full_like(rot_x, 0.0)
                    rot_z = np.full_like(rot_x, 0) 
                    rot_x = np.where(rot_x < -2.8, -3.145,
                                  np.where(rot_x > 2.8,  3.145, 0.0))
                    desired_rot = np.stack([rot_x,rot_y,rot_z],axis=1)
                    raw_action[k] = np.concatenate([xyz, desired_rot, v], axis=1)  # (B,26)

                elif "right" in k:
                    if script_xy:
                        xyz[:, 1] = target_right_y
                        xyz[:, 0] = target_right_x
                    else:
                        xyz[:, 1] = target_right_y

                    right_q = (self.agent.agents[1].robot.pose.inv() * self.agent.agents[1].ee_pose).q.cpu().numpy()
                    curr_eulers = np.stack([quat2euler(q) for q in right_q], axis=0)
                    rot_x = curr_eulers[:,0]
                    rot_y = np.full_like(rot_x,0.0)
                    rot_z = np.full_like(rot_x, 0)
                    rot_x = np.where(rot_x < -2.8, -3.145,
                                  np.where(rot_x > 2.8,  3.145, 0.0))
                    desired_rot = np.stack([rot_x,rot_y,rot_z],axis=1)

                    raw_action[k] = np.concatenate([xyz, desired_rot, v], axis=1)

                else:
                    raise NotImplementedError

        return raw_action

    def _get_obs_extra(self, info: Dict):
        n_envs = info['piano_activation'].shape[0]
        
        obs = dict(
            piano_activation=info['piano_activation'],
            piano_goal=info['goal_state'].reshape(-1).repeat(n_envs, 1), 
        )

        fingers = self.fingering_state.reshape(1, -1).repeat(n_envs, 1)
        obs['active_fingers'] = fingers
        return obs

    def evaluate(self):
        info = {}
        info.update(self._evaluate_piano_status())

        reward, reward_info = self.reward_fn(info)
        info['reward'] = reward
        info['reward_log'] = reward_info
        info['log'] = dict()

        self.colorize_keys(info)
        self._update_visual_effect(info)
        self.t_idx += 1
        self.key_presses.append(info['piano_activation'])
        return info

    def compute_dense_reward(self, obs: Any, action: Union[torch.Tensor, Dict], info: Dict):
        reward = info['reward']
        if self.coef_action_l1 != 0:
            if isinstance(action, dict):
                action_l1_rew = 0
                for v in action.values():
                    action_l1_rew = action_l1_rew - torch.linalg.norm(v, ord=1, dim=-1) * self.coef_action_l1
            else:
                action_l1_rew = -torch.linalg.norm(action, ord=1, dim=-1) * self.coef_action_l1
            reward = reward + action_l1_rew
            info['reward_log']['action_l1'] = action_l1_rew
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 1.0

    def _evaluate_piano_status(self):
        self._update_fingering_state() 

        goal_state = self.get_goal_state()
        if goal_state is not None:
            goal_current = goal_state[0]
        else:
            goal_current = torch.zeros(consts.NUM_KEYS + 1, device=self.device)

        if self.reward_fingering:
            self._rh_keys_current = self._rh_keys
            self._lh_keys_current = self._lh_keys

        piano_state = self.piano.get_qpos()
        piano_activation = (piano_state > self._key_press_threshold).float()

        info = dict(
            rh_keys_current=self._rh_keys_current, 
            lh_keys_current=self._lh_keys_current,
            goal_state=goal_state,
            goal_current=goal_current,
            piano_state=piano_state,
            piano_activation=piano_activation,
            piano_poses=self.piano_poses,
        )

        l_finger_pose = self.agent.agents[0].finger_tip_pose
        r_finger_pose = self.agent.agents[1].finger_tip_pose
        
        info['left_hand'] = []
        for i, finger in enumerate(l_finger_pose):
            adj_p = finger.p.clone()
            adj_p[:, 0] -= consts.DX_FINGERS[i]
            info['left_hand'].append(adj_p)
        info['right_hand'] = []
        for i, finger in enumerate(r_finger_pose):
            adj_p = finger.p.clone()
            adj_p[:, 0] -= consts.DX_FINGERS[i]
            info['right_hand'].append(adj_p)
        return info

    def get_eval_episode_metrics(self):
        """Compute metrics at the end of the episode for evaluation."""
        ground_truth = []
        for notes in self._notes:
            presses = np.zeros((consts.NUM_KEYS,), dtype=np.float64)
            keys = [note.key for note in notes]
            presses[keys] = 1.0
            ground_truth.append(presses)

        # align lengths
        min_len = min(len(ground_truth), len(self.key_presses))
        if min_len == 0:
            return dict(
                precision=np.array([0.0]),
                recall=np.array([0.0]),
                f1=np.array([0.0]),
                len_f1=np.array([0])
            )
        ground_truth = ground_truth[:min_len]
        key_presses = self.key_presses[:min_len]

        if min_len == 0:
            return dict()

        # Stack predictions: List of T (B, 88) -> (T, B, 88) -> (B, T, 88)
        y_preds = torch.stack(key_presses, dim=0).permute(1, 0, 2).cpu().numpy()
        B, T, K = y_preds.shape
        y_preds_flat = y_preds.reshape(B, -1)

        # Stack ground truth: List of T (88,) -> (T, 88) -> (1, T, 88) -> repeat -> (B, T, 88)
        y_trues = np.stack(ground_truth, axis=0) # (T, 88)
        y_trues = np.expand_dims(y_trues, axis=0) # (1, T, 88)
        y_trues = np.repeat(y_trues, B, axis=0) # (B, T, 88)
        y_trues_flat = y_trues.reshape(B, -1)

        f1s = []
        precisions = []
        recalls = []

        for i in range(B):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=y_trues_flat[i], y_pred=y_preds_flat[i], average="binary", zero_division=1
            )
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)

        return dict(
            precision=np.array(precisions).reshape(-1, 1),
            recall=np.array(recalls).reshape(-1, 1),
            f1=np.array(f1s).reshape(-1, 1),
            len_f1=np.array([min_len])
        )

    def _update_fingering_state(self) -> None:
        if self.t_idx >= len(self._notes):
            self.t_idx = len(self._notes) - 1

        fingering = [note.fingering for note in self._notes[self.t_idx]]
        fingering_keys = [note.key for note in self._notes[self.t_idx]]

        self._rh_keys: List[Tuple[int, int]] = []
        self._lh_keys: List[Tuple[int, int]] = []
        for key, finger in enumerate(fingering):
            piano_key = fingering_keys[key]
            if finger < 5:
                self._rh_keys.append((piano_key, finger))
            else:
                self._lh_keys.append((piano_key, finger - 5))

        if self.n_fingering_lookahead > 0:
            self.fingering_state = self.get_fingering_state(self.n_fingering_lookahead)
        else:
            self.fingering_state = torch.zeros((2, 5), dtype=torch.float32, device=self.device)
            for hand, keys in enumerate([self._rh_keys, self._lh_keys]):
                for key, mjcf_fingering in keys:
                    self.fingering_state[hand, mjcf_fingering] = 1.0

    def get_fingering_state(self, lookahead: int) -> None:
        fingering_state = torch.zeros((lookahead + 1, 2, 5), dtype=torch.float32, device=self.device)
        for t in range(lookahead + 1):
            t_finger = min(self.t_idx + t, len(self._notes) - 1)
            fingering = [note.fingering for note in self._notes[t_finger]]
            fingering_keys = [note.key for note in self._notes[t_finger]]

            _rh_keys: List[Tuple[int, int]] = []
            _lh_keys: List[Tuple[int, int]] = []
            for key, finger in enumerate(fingering):
                piano_key = fingering_keys[key]
                if finger < 5:
                    _rh_keys.append((piano_key, finger))
                else:
                    _lh_keys.append((piano_key, finger - 5))

            
            for hand, keys in enumerate([_rh_keys, _lh_keys]):
                for key, mjcf_fingering in keys:
                    fingering_state[t, hand, mjcf_fingering] = 1.0
        return fingering_state 

    def get_goal_state(self):
        if self.t_idx == len(self._notes):
            return

        goal_state = torch.zeros(
            (self.n_steps_lookahead + 1, consts.NUM_KEYS + 1),
            dtype=torch.float32, 
            device=self.device
        )
        t_start = self.t_idx
        t_end = min(t_start + (self.n_steps_lookahead + 1), len(self._notes))
        for i, t in enumerate(range(t_start, t_end)):
            keys = [note.key for note in self._notes[t]]
            goal_state[i, keys] = 1.0
            goal_state[i, -1] = self._sustains[t] 
        return goal_state

    @property
    def wrist_trajectory_left(self):
        if self._wrist_trajectory_left is None:
            self._compute_wrist_trajectory()
        return self._wrist_trajectory_left

    @property
    def wrist_trajectory_right(self):
        if self._wrist_trajectory_right is None:
            self._compute_wrist_trajectory()
        return self._wrist_trajectory_right

    def _compute_wrist_trajectory(self):
        """
        Compute the wrist trajectory based on the notes.
        """
        if self._piano_poses is None:
            _ = self.piano_poses

        piano_poses_np = self.piano_poses.cpu().numpy() 
        key_poses = piano_poses_np[0, :, 1]

        agent_left = self.agent.agents[0]
        agent_right = self.agent.agents[1]
        init_left_y = agent_left.ee_pose.p[0][1].item()
        init_right_y = agent_right.ee_pose.p[0][1].item()

        valid_indices_left = [0]
        valid_values_left_y = [init_left_y]
        valid_values_left_x = [0.615] 
        valid_indices_right = [0]
        valid_values_right_y = [init_right_y]
        valid_values_right_x = [0.615] 

        for t, notes in enumerate(self._notes):
            wrist_targets_left_y = []
            wrist_targets_left_x = []
            
            wrist_targets_right_y = []
            wrist_targets_right_x = []

            for note in notes:
                key_y = key_poses[note.key]
                if note.key in consts.BLACK_TWIN_KEY_INDICES or note.key in consts.BLACK_TRIPLET_KEY_INDICES:
                    target_x = 0.645 
                else:
                    target_x = 0.615

                # Left Hand: fingering 5-9 (mapped to 0-4)
                # Right Hand: fingering 0-4
                if note.fingering < 5: # Right Hand
                    finger_idx = note.fingering
                    offset = consts.DY_FINGERS_R[finger_idx]
                    target_wrist = key_y - offset
                    wrist_targets_right_y.append(target_wrist)
                    wrist_targets_right_x.append(target_x)
                else: # Left Hand
                    finger_idx = note.fingering - 5
                    offset = consts.DY_FINGERS_L[finger_idx]
                    target_wrist = key_y - offset
                    wrist_targets_left_y.append(target_wrist)
                    wrist_targets_left_x.append(target_x)

            # Average targets
            if len(wrist_targets_left_y) > 0:
                avg_left_y = np.mean(wrist_targets_left_y)
                avg_left_x = np.min(wrist_targets_left_x)
                
                valid_indices_left.append(t)
                valid_values_left_y.append(avg_left_y)
                valid_values_left_x.append(avg_left_x)

            if len(wrist_targets_right_y) > 0:
                avg_right_y = np.mean(wrist_targets_right_y)
                avg_right_x = np.min(wrist_targets_right_x) 
                
                valid_indices_right.append(t)
                valid_values_right_y.append(avg_right_y)
                valid_values_right_x.append(avg_right_x)

        # Interpolate
        timesteps = np.arange(len(self._notes))

        self._wrist_trajectory_left = np.zeros((len(self._notes), 2))
        self._wrist_trajectory_right = np.zeros((len(self._notes), 2))

        if len(valid_indices_left) > 0:
            self._wrist_trajectory_left[:, 0] = np.interp(timesteps, valid_indices_left, valid_values_left_x)
            self._wrist_trajectory_left[:, 1] = np.interp(timesteps, valid_indices_left, valid_values_left_y)
        else:
            self._wrist_trajectory_left[:, 0] = 0.615
            self._wrist_trajectory_left[:, 1] = 0.0

        if len(valid_indices_right) > 0:
            self._wrist_trajectory_right[:, 0] = np.interp(timesteps, valid_indices_right, valid_values_right_x)
            self._wrist_trajectory_right[:, 1] = np.interp(timesteps, valid_indices_right, valid_values_right_y)
        else:
            self._wrist_trajectory_right[:, 0] = 0.615
            self._wrist_trajectory_right[:, 1] = 0.0

        # Hardcoded because I know the orientation of the panda 
        self._wrist_trajectory_left[:, 1] = self._wrist_trajectory_left[:, 1] - agent_left.keyframes[self.init_keyframe].pose.p[1]
        self._wrist_trajectory_right[:, 1] = self._wrist_trajectory_right[:, 1] - agent_right.keyframes[self.init_keyframe].pose.p[1]


    @property 
    def piano_xyz_th(self):
        if self._piano_xyz_th is None:
            self._piano_xyz_th = torch.tensor(self.piano_xyz, device=self.device).reshape(1, 3)
        return self._piano_xyz_th

    @property
    def piano_poses(self):
        if self._piano_poses is None:
            piano_poses = dict()
            for key, val in self.piano.links_map.items():
                if key == "base_link":
                    continue 
                key_num = int(key.split("_")[-1])
                key_pose = val.pose.p.clone()
                if key_num in consts.WHITE_KEY_INDICES:
                    height = consts.WHITE_KEY_HEIGHT
                    length = consts.WHITE_KEY_LENGTH
                    key_pose[:, 2] += height
                    key_pose[:, 0] -= length * 0.65
                elif key_num in consts.BLACK_TWIN_KEY_INDICES or key_num in consts.BLACK_TRIPLET_KEY_INDICES:
                    height = consts.BLACK_KEY_HEIGHT
                    length = consts.BLACK_KEY_LENGTH
                    key_pose[:, 2] += height
                    key_pose[:, 0] -= length * 0.8
                else:
                    raise NotImplementedError
                piano_poses[key_num] = key_pose 
            piano_poses_th = [piano_poses[key].to(self.device) for key in sorted(piano_poses.keys())]
            piano_poses_th = torch.stack(piano_poses_th, dim=0).permute(1, 0, 2)  # (B, N_KEYS, 7)
            self._piano_poses = piano_poses_th
        return self._piano_poses
    

    def colorize_keys(self, info):
        # reset all keys
        for key in consts.WHITE_KEY_INDICES:
            link_name = "white_key_" + str(key)
            self.piano.links_map[link_name].render_shapes[0][0].material.set_base_color((0.9, 0.9, 0.9, 1.0))
        for key in consts.BLACK_TWIN_KEY_INDICES + consts.BLACK_TRIPLET_KEY_INDICES:
            link_name = "black_key_" + str(key)
            self.piano.links_map[link_name].render_shapes[0][0].material.set_base_color((0.1, 0.1, 0.1, 1.0))

        # for now only works with one env! so env_idx is not used. 
        activation = info['piano_activation']
        active_key_indices = torch.nonzero(activation)
        key_should_pressed = torch.nonzero(info['goal_current'][:-1]).flatten().cpu().numpy() # this is suboptimal because cuda. but better for debug...
        for active in active_key_indices:
            env_idx, key_idx = active[0].cpu().item(), active[1].cpu().item()
            if env_idx != 0:
                continue
            if key_idx in consts.WHITE_KEY_INDICES:
                link_name = "white_key_" + str(key_idx)
            elif key_idx in consts.BLACK_TWIN_KEY_INDICES or key_idx in consts.BLACK_TRIPLET_KEY_INDICES:
                link_name = "black_key_" + str(key_idx)
            else:
                raise NotImplementedError
            if key_idx in key_should_pressed:
                # Correctly pressed keys = Green
                self.piano.links_map[link_name].render_shapes[0][0].material.set_base_color((0.0, 1.0, 0.0, 1.0))
            else:
                # Wrongly pressed keys = Red
                self.piano.links_map[link_name].render_shapes[0][0].material.set_base_color((1.0, 0.0, 0.0, 1.0))
        for key in key_should_pressed:
            if key not in active_key_indices:
                # Unpressed keys = Yellow
                if key in consts.WHITE_KEY_INDICES:
                    link_name = "white_key_" + str(key)
                elif key in consts.BLACK_TWIN_KEY_INDICES or key in consts.BLACK_TRIPLET_KEY_INDICES:
                    link_name = "black_key_" + str(key)
                else:
                    raise NotImplementedError
                self.piano.links_map[link_name].render_shapes[0][0].material.set_base_color((1.0, 1.0, 0.0, 1.0))

    def _update_visual_effect(self, info):
        """Update the trajectory marker and the visual effects for valid hits."""
        
        # Left Hand
        keys_current_l = info.get('lh_keys_current', [])
        for (key, finger) in keys_current_l:
            key_pos = info['piano_poses'][0, key, :3].clone()
            self.lh_key_markers[finger].set_pose(sapien.Pose(p=key_pos.cpu().numpy()))

        # Right Hand
        keys_current_r = info.get('rh_keys_current', [])
        for (key, finger) in keys_current_r:
            key_pos = info['piano_poses'][0, key, :3].clone()
            self.rh_key_markers[finger].set_pose(sapien.Pose(p=key_pos.cpu().numpy()))
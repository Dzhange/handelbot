import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
import wandb
from tensordict import TensorDict, from_module
from tensordict.nn import CudaGraphModule, TensorDictModule
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from envs import FlattenPianoActionSpaceWrapper, RecordPianoEpisode
from mani_skill.utils import gym_utils
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Piano2"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "PPO"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""

    # Environment specific arguments
    env_id: str = "Piano-Bimanual-v1"
    """the id of the environment"""
    env_vectorization: str = "gpu"
    """the type of environment vectorization to use"""
    robot_uids: Optional[str] = "delto_left_panda_reduced_fixed1,delto_right_panda_reduced_fixed1"
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 9
    """the number of parallel evaluation environments"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """how often to reconfigure the eval environment"""
    eval_freq: int = 100
    """evaluation frequency in terms of iterations"""
    save_freq: int = 100
    """model saving frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = "pd_ee_script_xy_pos_joint_delta_pos"
    """the control mode to use for the environment"""

    # Piano Env Specifics
    coef_key: float = 1.0
    """coefficient of the key pressing reward"""
    coef_finger: float = 1.0
    """coefficient of the finger distance penalty"""
    coef_action_l1: float = 0.01
    """coefficient of the action L1 norm reward"""
    key_on: float = 0.7
    """threshold for key activation in reward calculation"""
    midi_name: Optional[str] = None
    """name of the MIDI file to play"""
    piano_xyz: Tuple[float, float, float] = (0.965, 0, 0.145)
    """base XYZ position of the piano"""
    note_trajectory: Optional[str] = None
    """path to a pre-computed note trajectory pickle file"""
    horizon: int = 150
    """maximum number of steps per episode"""
    domain_rand_gains: bool = False
    """whether to use domain randomization for gains"""
    domain_rand_keys: bool = False
    """whether to use domain randomization for piano keys"""
    domain_rand_scene: bool = False
    """whether to use domain randomization for scene (piano height, hand pose)"""

    # Algorithm specific arguments
    total_timesteps: int = 40000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 8
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    finite_horizon_gae: bool = False

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

    # Torch optimizations
    compile: bool = False
    """whether to use torch.compile."""
    cudagraphs: bool = True
    """whether to use cudagraphs on top of compile."""
    control_freq: int = 10
    

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, n_obs, n_act, device=None):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1, device=device)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(n_obs, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256, device=device)),
            nn.Tanh(),
            layer_init(nn.Linear(256, n_act, device=device), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, n_act, device=device))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, obs, action=None):
        action_mean = self.actor_mean(obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std, validate_args=False)
        if action is None:
            action = action_mean + action_std * torch.randn_like(action_mean)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

def gae(next_obs, next_done, container, final_values):
    # bootstrap value if not done
    next_value = get_value(next_obs).reshape(-1)
    lastgaelam = 0
    nextnonterminals = (~container["dones"]).float().unbind(0)
    vals = container["vals"]
    vals_unbind = vals.unbind(0)
    rewards = container["rewards"].unbind(0)

    advantages = []
    nextnonterminal = (~next_done).float()
    nextvalues = next_value
    for t in range(args.num_steps - 1, -1, -1):
        cur_val = vals_unbind[t]
        # real_next_values = nextvalues * nextnonterminal
        real_next_values = nextnonterminal * nextvalues + final_values[t] # t instead of t+1
        delta = rewards[t] + args.gamma * real_next_values - cur_val
        advantages.append(delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
        lastgaelam = advantages[-1]

        nextnonterminal = nextnonterminals[t]
        nextvalues = cur_val

    advantages = container["advantages"] = torch.stack(list(reversed(advantages)))
    container["returns"] = advantages + vals
    return container


def rollout(obs, done):
    ts = []
    final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
    for step in range(args.num_steps):
        # ALGO LOGIC: action logic
        action, logprob, _, value = policy(obs=obs)
        
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, next_done, infos = step_func(action)

        # --- log training stats ---            
        for k in infos["log"].keys():
            logger.add_scalar(f"train_log/{k}", infos["log"][k].float().mean(), global_step)
        
        for k in infos["reward_log"].keys():
            logger.add_scalar(f"train_reward_log/{k}", infos["reward_log"][k].float().mean(), global_step)

        if "final_info" in infos:
            final_info = infos["final_info"]
            done_mask = infos["_final_info"]
            # --- log final info ---
            for k, v in final_info["episode"].items():
                logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
        
            with torch.no_grad():
                final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"][done_mask]).view(-1)

        ts.append(
            TensorDict._new_unsafe(
                obs=obs,
                # cleanrl ppo examples associate the done with the previous obs (not the done resulting from action)
                dones=done,
                vals=value.flatten(),
                actions=action,
                logprobs=logprob,
                rewards=reward,
                batch_size=(args.num_envs,),
            )
        )
        # NOTE (stao): change here for gpu env
        obs = next_obs = next_obs
        done = next_done
    # NOTE (stao): need to do .to(device) i think? otherwise container.device is None, not sure if this affects anything
    container = torch.stack(ts, 0).to(device)
    return next_obs, done, container, final_values


def update(obs, actions, logprobs, advantages, returns, vals):
    optimizer.zero_grad()
    _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs, actions)
    logratio = newlogprob - logprobs
    ratio = logratio.exp()

    with torch.no_grad():
        # calculate approx_kl http://joschu.net/blog/kl-approx.html
        old_approx_kl = (-logratio).mean()
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean()

    if args.norm_adv:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    # Value loss
    newvalue = newvalue.view(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - returns) ** 2
        v_clipped = vals + torch.clamp(
            newvalue - vals,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

    entropy_loss = entropy.mean()
    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    loss.backward()
    gn = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
    optimizer.step()

    return approx_kl, v_loss.detach(), pg_loss.detach(), entropy_loss.detach(), old_approx_kl, clipfrac, gn


update = TensorDictModule(
    update,
    in_keys=["obs", "actions", "logprobs", "advantages", "returns", "vals"],
    out_keys=["approx_kl", "v_loss", "pg_loss", "entropy_loss", "old_approx_kl", "clipfrac", "gn"],
)

if __name__ == "__main__":
    args = tyro.cli(Args)
    print("Arguments:", args)
    # if not args.evaluate: exit()

    batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = batch_size // args.num_minibatches
    args.batch_size = args.num_minibatches * args.minibatch_size
    args.num_iterations = args.total_timesteps // args.batch_size
    # run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{datetime.now().strftime('%m-%d_%H-%M')}"
    run_name = args.exp_name


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ####### Environment setup #######
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    if args.robot_uids is not None:
        env_kwargs["robot_uids"] = tuple(args.robot_uids.split(","))
    envs = gym.make(
        args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs, 
        coef_key=args.coef_key,
        coef_finger=args.coef_finger,
        coef_action_l1=args.coef_action_l1,
        key_on=args.key_on,
        midi_name=args.midi_name,
        note_trajectory_name=args.note_trajectory,
        piano_xyz=args.piano_xyz,
        max_episode_steps=args.horizon,
        horizon=args.horizon,
        domain_rand_gains=args.domain_rand_gains,
        domain_rand_keys=args.domain_rand_keys,
        domain_rand_scene=args.domain_rand_scene,
        is_eval=False,
        sim_config=dict(control_freq=args.control_freq)
    )
    eval_envs = gym.make(
        args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, 
        human_render_camera_configs=dict(shader_pack="default"), **env_kwargs, 
        coef_key=args.coef_key,
        coef_finger=args.coef_finger,
        coef_action_l1=args.coef_action_l1,
        key_on=args.key_on,
        midi_name=args.midi_name,
        note_trajectory_name=args.note_trajectory,
        piano_xyz=args.piano_xyz,
        max_episode_steps=args.horizon,
        horizon=args.horizon,
        domain_rand_gains=args.domain_rand_gains,
        domain_rand_keys=args.domain_rand_keys,
        domain_rand_scene=args.domain_rand_scene,
        is_eval=True,
        sim_config=dict(control_freq=args.control_freq)
    )
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenPianoActionSpaceWrapper(envs, args.control_mode, False)
        eval_envs = FlattenPianoActionSpaceWrapper(eval_envs, args.control_mode, False )
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordPianoEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=args.control_freq, info_on_video=True)
        eval_envs = RecordPianoEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory, save_video=args.capture_video, trajectory_name="trajectory", max_steps_per_video=args.horizon, video_fps=args.control_freq, info_on_video=True)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=args.horizon, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=args.horizon, partial_reset=False)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["ppo", "walltime_efficient", f"GPU:{torch.cuda.get_device_name()}"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")
    n_act = math.prod(envs.single_action_space.shape)
    if "ee_script" in args.control_mode:
        n_act -= 12
    n_obs = math.prod(envs.single_observation_space.shape)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # Register step as a special op not to graph break
    # @torch.library.custom_op("mylib::step", mutates_args=())
    def step_func(action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE (stao): change here for gpu env
        next_obs, reward, terminations, truncations, info = envs.step(action)
        next_done = torch.logical_or(terminations, truncations)
        return next_obs, reward, next_done, info

    ####### Agent #######
    agent = Agent(n_obs, n_act, device=device)
    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))
    # Make a version of agent with detached params
    agent_inference = Agent(n_obs, n_act, device=device)
    agent_inference_p = from_module(agent).data
    agent_inference_p.to_module(agent_inference)

    ####### Optimizer #######
    optimizer = optim.Adam(
        agent.parameters(),
        lr=torch.tensor(args.learning_rate, device=device),
        eps=1e-5,
        capturable=args.cudagraphs and not args.compile,
    )

    ####### Executables #######
    # Define networks: wrapping the policy in a TensorDictModule allows us to use CudaGraphModule
    policy = agent_inference.get_action_and_value
    get_value = agent_inference.get_value

    # Compile policy
    if args.compile:
        policy = torch.compile(policy)
        gae = torch.compile(gae, fullgraph=True)
        update = torch.compile(update)

    if args.cudagraphs:
        policy = CudaGraphModule(policy)
        gae = CudaGraphModule(gae)
        update = CudaGraphModule(update)

    global_step = 0
    start_time = time.time()
    container_local = None
    next_obs = envs.reset()[0]
    next_done = torch.zeros(args.num_envs, device=device, dtype=torch.bool)
    pbar = tqdm.tqdm(range(1, args.num_iterations + 1))

    cumulative_times = defaultdict(float)
    best_eval_f1 = -1.0

    for iteration in pbar:
        agent.eval()
        if iteration % args.eval_freq == 1:
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0

            all_obs = [eval_obs]
            all_actions = []
            all_piano_activation = []
            all_piano_goal = []
            all_left_hand = []
            all_right_hand = []
            all_active_fingers = []

            all_left_hand_qpos = []
            all_left_hand_target_qpos = []
            all_left_arm_qpos = []
            all_left_arm_target_qpos = []
            all_right_hand_qpos = []
            all_right_hand_target_qpos = []
            all_right_arm_qpos = []
            all_right_arm_target_qpos = []
            all_left_arm_ee_pose = []
            all_left_arm_target_ee_pose = []
            all_right_arm_ee_pose = []
            all_right_arm_target_ee_pose = []

            for idx in range(args.horizon):
                with torch.no_grad():
                    action = agent.actor_mean(eval_obs)
                    all_actions.append(action)
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(action)
                    all_obs.append(eval_obs)

                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)

                    # Collect extra observations for saving
                    env = eval_envs._env
                    # Left Agent (0)
                    agent_l = env.agent.agents[0]
                    if 'hand' in agent_l.controller.controllers:
                        c = agent_l.controller.controllers['hand']
                        all_left_hand_qpos.append(c.qpos)
                        all_left_hand_target_qpos.append(c._target_qpos)

                    if 'arm' in agent_l.controller.controllers:
                        c = agent_l.controller.controllers['arm']
                        all_left_arm_qpos.append(c.qpos)
                        all_left_arm_target_qpos.append(c._target_qpos)
                        all_left_arm_ee_pose.append(c.ee_pose_at_base.raw_pose)
                        all_left_arm_target_ee_pose.append(c._target_pose.raw_pose)

                    # Right Agent (1)
                    agent_r = env.agent.agents[1]
                    if 'hand' in agent_r.controller.controllers:
                        c = agent_r.controller.controllers['hand']
                        all_right_hand_qpos.append(c.qpos)
                        all_right_hand_target_qpos.append(c._target_qpos)

                    if 'arm' in agent_r.controller.controllers:
                        c = agent_r.controller.controllers['arm']
                        all_right_arm_qpos.append(c.qpos)
                        all_right_arm_target_qpos.append(c._target_qpos)
                        all_right_arm_ee_pose.append(c.ee_pose_at_base.raw_pose)
                        all_right_arm_target_ee_pose.append(c._target_pose.raw_pose)

                    fingers = env.fingering_state 
                    active_fingers = fingers.reshape(-1) 

                    goal = eval_infos['goal_state']

                    lh = torch.hstack([p.p.unsqueeze(1) for p in env.agent.agents[0].finger_tip_pose])
                    lh = (lh - env.piano_xyz_th).reshape(-1, 5*3) 

                    rh = torch.hstack([p.p.unsqueeze(1) for p in env.agent.agents[1].finger_tip_pose])
                    rh = (rh - env.piano_xyz_th).reshape(-1, 5*3) 
                    all_active_fingers.append(active_fingers)
                    all_piano_goal.append(goal)
                    all_left_hand.append(lh)
                    all_right_hand.append(rh)
                    all_piano_activation.append(eval_infos['piano_activation'])

            # Record data at the second-to-last step to prevent it from being flushed.
            # This works only when partial_reset=False, which ensures full episodes run to completion regardless of success or failure.
            eval_metrics.update(eval_envs._env.eval_episode_metrics)

            env_f1s = eval_metrics['f1'].flatten()
            max_f1 = np.max(env_f1s)
            best_env_idx = np.argmax(env_f1s)

            if max_f1 > best_eval_f1:
                best_eval_f1 = max_f1
                print(f"New best F1 score: {best_eval_f1} (Env {best_env_idx})")

                if args.capture_video or args.save_trajectory:
                    best_obs = np.array(torch.stack(all_obs)[:, best_env_idx].cpu().numpy())
                    np.save(os.path.join(eval_output_dir, "all_obs.npy"), best_obs)

                    best_actions = np.array(torch.stack(all_actions)[:, best_env_idx].cpu().numpy())
                    np.save(os.path.join(eval_output_dir, "all_actions.npy"), best_actions)

                    best_piano_activation = np.array(torch.stack(all_piano_activation)[:, best_env_idx].cpu().numpy())
                    np.save(os.path.join(eval_output_dir, "piano_activation.npy"), best_piano_activation)

                    best_piano_goal = np.array(torch.stack(all_piano_goal)[:, best_env_idx].cpu().numpy())
                    np.save(os.path.join(eval_output_dir, "piano_goal.npy"), best_piano_goal)

                    best_active_fingers = np.array(torch.stack(all_active_fingers)[:, best_env_idx].cpu().numpy()) # (T, flattened_fingers)
                    np.save(os.path.join(eval_output_dir, "active_fingers.npy"), best_active_fingers)

                    best_left_hand = np.array(torch.stack(all_left_hand)[:, best_env_idx].cpu().numpy())
                    np.save(os.path.join(eval_output_dir, "left_hand.npy"), best_left_hand)

                    best_right_hand = np.array(torch.stack(all_right_hand)[:, best_env_idx].cpu().numpy())
                    np.save(os.path.join(eval_output_dir, "right_hand.npy"), best_right_hand)

                    if len(all_left_hand_qpos) > 0:
                        best_lh_qpos = np.array(torch.stack(all_left_hand_qpos)[:, best_env_idx].cpu().numpy())
                        np.save(os.path.join(eval_output_dir, "left_hand_qpos.npy"), best_lh_qpos)

                        best_lh_target_qpos = np.array(torch.stack(all_left_hand_target_qpos)[:, best_env_idx].cpu().numpy())
                        np.save(os.path.join(eval_output_dir, "left_hand_target_qpos.npy"), best_lh_target_qpos)

                    if len(all_left_arm_qpos) > 0:
                        best_la_qpos = np.array(torch.stack(all_left_arm_qpos)[:, best_env_idx].cpu().numpy())
                        np.save(os.path.join(eval_output_dir, "left_arm_qpos.npy"), best_la_qpos)

                        best_la_target_qpos = np.array(torch.stack(all_left_arm_target_qpos)[:, best_env_idx].cpu().numpy())
                        np.save(os.path.join(eval_output_dir, "left_arm_target_qpos.npy"), best_la_target_qpos)

                        best_la_ee_pose = np.array(torch.stack(all_left_arm_ee_pose)[:, best_env_idx].cpu().numpy())
                        np.save(os.path.join(eval_output_dir, "left_arm_ee_pose.npy"), best_la_ee_pose)

                        best_la_target_ee_pose = np.array(torch.stack(all_left_arm_target_ee_pose)[:, best_env_idx].cpu().numpy())
                        np.save(os.path.join(eval_output_dir, "left_arm_target_ee_pose.npy"), best_la_target_ee_pose)

                    if len(all_right_hand_qpos) > 0:
                        best_rh_qpos = np.array(torch.stack(all_right_hand_qpos)[:, best_env_idx].cpu().numpy())
                        np.save(os.path.join(eval_output_dir, "right_hand_qpos.npy"), best_rh_qpos)

                        best_rh_target_qpos = np.array(torch.stack(all_right_hand_target_qpos)[:, best_env_idx].cpu().numpy())
                        np.save(os.path.join(eval_output_dir, "right_hand_target_qpos.npy"), best_rh_target_qpos)

                    if len(all_right_arm_qpos) > 0:
                        best_ra_qpos = np.array(torch.stack(all_right_arm_qpos)[:, best_env_idx].cpu().numpy())
                        np.save(os.path.join(eval_output_dir, "right_arm_qpos.npy"), best_ra_qpos)

                        best_ra_target_qpos = np.array(torch.stack(all_right_arm_target_qpos)[:, best_env_idx].cpu().numpy())
                        np.save(os.path.join(eval_output_dir, "right_arm_target_qpos.npy"), best_ra_target_qpos)

                        best_ra_ee_pose = np.array(torch.stack(all_right_arm_ee_pose)[:, best_env_idx].cpu().numpy())
                        np.save(os.path.join(eval_output_dir, "right_arm_ee_pose.npy"), best_ra_ee_pose)

                        best_ra_target_ee_pose = np.array(torch.stack(all_right_arm_target_ee_pose)[:, best_env_idx].cpu().numpy())
                        np.save(os.path.join(eval_output_dir, "right_arm_target_ee_pose.npy"), best_ra_target_ee_pose)

                    np.set_printoptions(suppress=True, precision=3)
                    with open(os.path.join(eval_output_dir, "best_f1_score.txt"), "w") as f:
                        f.write(str(best_eval_f1))

            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean() if isinstance(v[0], torch.Tensor) else np.mean(v)
                eval_metrics_mean[k] = mean
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
            pbar.set_description(
                f"return: {eval_metrics_mean['return']:.2f}"
            )
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
        if args.save_model and iteration % args.save_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"].copy_(lrnow)

        torch.compiler.cudagraph_mark_step_begin()
        rollout_time = time.perf_counter()
        next_obs, next_done, container, final_values = rollout(next_obs, next_done)
        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        global_step += container.numel()

        update_time = time.perf_counter()
        container = gae(next_obs, next_done, container, final_values)
        container_flat = container.view(-1)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            b_inds = torch.randperm(container_flat.shape[0], device=device).split(args.minibatch_size)
            for b in b_inds:
                container_local = container_flat[b]

                out = update(container_local, tensordict_out=TensorDict())
                clipfracs.append(out["clipfrac"])
                if args.target_kl is not None and out["approx_kl"] > args.target_kl:
                    break
            else:
                continue
            break
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("charts/iteration", iteration, global_step)
        logger.add_scalar("losses/value_loss", out["v_loss"].item(), global_step)
        logger.add_scalar("losses/policy_loss", out["pg_loss"].item(), global_step)
        logger.add_scalar("losses/entropy", out["entropy_loss"].item(), global_step)
        logger.add_scalar("losses/old_approx_kl", out["old_approx_kl"].item(), global_step)
        logger.add_scalar("losses/approx_kl", out["approx_kl"].item(), global_step)
        logger.add_scalar("losses/clipfrac", torch.stack(clipfracs).mean().cpu().item(), global_step)
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
    if not args.evaluate:
        if args.save_model:
            model_path = f"runs/{run_name}/final_ckpt.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        logger.close()
    envs.close()
    eval_envs.close()

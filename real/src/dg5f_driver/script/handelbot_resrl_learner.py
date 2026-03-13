from collections import defaultdict
import os
import random
import time
from typing import Optional
import threading

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro
import yaml
import wandb
import zmq

from real.src.dg5f_driver.utils.rl.replay_buffer import ReplayBuffer
from real.src.dg5f_driver.utils.rl.td3 import Actor, QNetwork, Logger

class SimpleNamespace:
    def __init__(self, dictionary, **kwargs):
        self.__dict__.update(dictionary)
        self.__dict__.update(kwargs)

def zmq_receiver(sock, rb_left, rb_right, training_status):
    while True:
        try:
            batch = sock.recv_pyobj() 
            avg_reward_left = None
            avg_reward_right = None
            if isinstance(batch, tuple):
                batch, avg_reward_left, avg_reward_right = batch

            if batch == "SHUTDOWN" or batch == "PAUSE":
                print(f"Received {batch} signal from actor. Pausing learner...")
                training_status['paused'] = True
                continue
            
            if batch == "RESUME":
                print("Received RESUME signal from actor. Resuming learner...")
                training_status['paused'] = False
                continue
            
            if training_status['paused']:
                print("Received data. Resuming learner...")
                training_status['paused'] = False

            for b in batch:
                t_left = {
                    'observations': b['obs_left'],
                    'actions': b['action_left'],
                    'next_observations': b['next_obs_left'],
                    'rewards': b['rew_left'][:2],
                    'masks': b['masks'],
                    'dones': b['dones']
                }
                rb_left.insert(t_left, avg_reward=avg_reward_left)
                
                t_right = {
                    'observations': b['obs_right'],
                    'actions': b['action_right'],
                    'next_observations': b['next_obs_right'],
                    'rewards': b['rew_right'][:2],
                    'masks': b['masks'],
                    'dones': b['dones']
                }
                rb_right.insert(t_right, avg_reward=avg_reward_right)
        except Exception as e:
            print(f"Error in zmq_receiver: {e}")

class Td3Agent:
    """Helper class to encapsulate one TD3 agent (Actor + Critics)"""
    def __init__(self, obs_dim, action_dim, action_high, policy_noise, device, args):
        self.args = args
        self.device = device
        self.n_critics = args.n_critics
        self.history_length = args.history_length
        
        if isinstance(args.target_noise, list):
            self.target_noise = torch.tensor(args.target_noise, device=device).float()
        else:
            self.target_noise = args.target_noise
            
        if isinstance(args.noise_clip, list):
            self.noise_clip = torch.tensor(args.noise_clip, device=device).float()
        else:
            self.noise_clip = args.noise_clip
            
        self.key_on_curriculum = args.key_on_curriculum

        if isinstance(policy_noise, list):
            policy_noise = torch.tensor(policy_noise, device=device).float()

        self.actor = Actor(obs_dim, action_dim, action_high, policy_noise=policy_noise, actor_hidden_dims=args.actor_hidden_dims, dropout=args.dropout, layernorm=args.layernorm, history_length=self.history_length, noise_beta=args.noise_beta).to(device)
        self.actor_target = Actor(obs_dim, action_dim, action_high, policy_noise=policy_noise, actor_hidden_dims=args.actor_hidden_dims, dropout=args.dropout, layernorm=args.layernorm, history_length=self.history_length, noise_beta=args.noise_beta).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=args.policy_lr)

        self.qfs = nn.ModuleList([QNetwork(obs_dim, action_dim, critic_hidden_dims=args.critic_hidden_dims, dropout=args.dropout, layernorm=args.layernorm, history_length=self.history_length).to(device) for _ in range(self.n_critics)])
        self.qfs_target = nn.ModuleList([QNetwork(obs_dim, action_dim, critic_hidden_dims=args.critic_hidden_dims, dropout=args.dropout, layernorm=args.layernorm, history_length=self.history_length).to(device) for _ in range(self.n_critics)])
        
        for qf, qf_target in zip(self.qfs, self.qfs_target):
            qf_target.load_state_dict(qf.state_dict())
            
        self.q_optimizer = optim.Adam(self.qfs.parameters(), lr=args.q_lr)

    def get_key_on(self, env_step):
        for key_on, n_steps in self.key_on_curriculum:
            if n_steps == -1 or env_step < n_steps:
                return key_on
        return self.key_on_curriculum[-1][0]

    def update(self, data, global_update, env_step, logger, prefix=""):
        key_on = self.get_key_on(env_step)
        rewards = key_on * data['rewards'][:, 0] + (1 - key_on) * data['rewards'][:, 1]# + data['rewards'][:, 2]
        
        metrics = {}
        if global_update % self.args.log_freq == 0:
            metrics[f"stats/{prefix}key_on"] = key_on

        if global_update % self.args.update_critic_every == 0:
            current_target_noise = self.target_noise
            current_noise_clip = self.noise_clip

            with torch.no_grad():
                next_state_actions = self.actor_target(data['next_observations'])
                noise = (torch.randn_like(next_state_actions) * current_target_noise).clamp(-current_noise_clip, current_noise_clip)
                
                if not isinstance(self.actor.action_scale, torch.Tensor):
                     action_scale = torch.tensor(self.actor.action_scale, device=self.device)
                else:
                     action_scale = self.actor.action_scale
                
                next_state_actions = (next_state_actions + noise).clamp(-action_scale, action_scale)
    
                next_state_actions = (next_state_actions + noise).clamp(-action_scale, action_scale)
    
                # Randomly subsample 2 indices for target calculation
                target_idxs = np.random.choice(self.n_critics, 2, replace=False)
                
                target_q_values = []
                for idx in target_idxs:
                    target_q_values.append(self.qfs_target[idx](data['next_observations'], next_state_actions))
                    
                min_qf_next_target = torch.min(torch.stack(target_q_values, dim=0), dim=0)[0]
                min_qf_next_target = torch.min(torch.stack(target_q_values, dim=0), dim=0)[0]
                next_q_value = rewards.flatten() + (data['masks'].flatten()) * self.args.gamma * (min_qf_next_target).view(-1)
    
            qf_loss = 0
            for qf in self.qfs:
                qf_a_values = qf(data['observations'], data['actions']).view(-1)
                qf_loss += F.mse_loss(qf_a_values, next_q_value)
    
            self.q_optimizer.zero_grad()
            qf_loss.backward()
            qf_grad_norm = torch.nn.utils.clip_grad_norm_(self.qfs.parameters(), max_norm=self.args.grad_clip)
            self.q_optimizer.step()
            
            # Metrics to log
            qf1_log = self.qfs[0](data['observations'], data['actions']) # tbh I should remove this lol
            metrics.update({
                f"losses/{prefix}qf1_values": qf1_log.mean().item(),
                f"losses/{prefix}qf1_values_max": qf1_log.max().item(),
                f"losses/{prefix}qf1_values_min": qf1_log.min().item(),
                f"losses/{prefix}qf_loss": qf_loss.item() / self.n_critics,
                f"losses/{prefix}target_q": next_q_value.mean().item(),
                f"losses/{prefix}qf_grad_norm": qf_grad_norm.item(),
            })
            metrics[f"losses/{prefix}actions"] = data['actions'].mean()

        # Update actor
        if global_update % self.args.update_actor_every == 0:
            pi = self.actor(data['observations'])
            
            critic_idxs = np.random.choice(self.n_critics, 2, replace=False)
            q_values = []
            for idx in critic_idxs:
                q_values.append(self.qfs[idx](data['observations'], pi))
            
            min_qf_pi = torch.min(torch.stack(q_values, dim=0), dim=0)[0]
            actor_loss = -min_qf_pi.mean()
 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.args.grad_clip)
            self.actor_optimizer.step()

            # Target updates
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            
            for qf, qf_target in zip(self.qfs, self.qfs_target):
                for param, target_param in zip(qf.parameters(), qf_target.parameters()):
                     target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)
            
            metrics.update({
                f"losses/{prefix}actor_loss": actor_loss.item(),
                f"losses/{prefix}actor_grad_norm": actor_grad_norm.item(),
            })
            
        metrics.update({
            f"stats/{prefix}reward": rewards.mean().item()
        })
        
        return metrics

    def state_dict(self):
        sd = {
            'actor': self.actor.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict()
        }
        for i, qf in enumerate(self.qfs):
             sd[f'qf{i+1}'] = qf.state_dict()
             sd[f'qf{i+1}_target'] = self.qfs_target[i].state_dict()
        return sd

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.q_optimizer.load_state_dict(state_dict['q_optimizer'])
        
        for i, qf in enumerate(self.qfs):
             key = f'qf{i+1}'
             if key in state_dict:
                 qf.load_state_dict(state_dict[key])
                 target_key = f'qf{i+1}_target'
                 if target_key in state_dict:
                      self.qfs_target[i].load_state_dict(state_dict[target_key])
                 else:
                      self.qfs_target[i].load_state_dict(qf.state_dict())


if __name__ == "__main__":
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "../config/handelbot_resrl.yaml")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    args = SimpleNamespace(config_data)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Replay Buffers
    rb_left = ReplayBuffer(
        obs_dim=args.obs_dim_left * args.history_length,
        action_dim=args.action_dim // 2,
        capacity=args.buffer_size, 
    )
    rb_right = ReplayBuffer(
        obs_dim=args.obs_dim_right * args.history_length,
        action_dim=args.action_dim // 2,
        capacity=args.buffer_size, 
    )

    training_status = {'paused': False}
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.bind("tcp://*:5588")
    
    receiver_thread = threading.Thread(
        target=zmq_receiver,
        args=(sock, rb_left, rb_right, training_status),
        daemon=True,
    )
    receiver_thread.start()

    sample_args_left = {"batch_size": args.batch_size}
    sample_args_right = {"batch_size": args.batch_size}
    replay_iterator_left = rb_left.get_iterator(
        sample_args=sample_args_left,
        device=device,
    )
    replay_iterator_right = rb_right.get_iterator(
        sample_args=sample_args_right,
        device=device,
    )

    # Logger
    print("Running training (Separate Agents)")
    if args.track:
        config = vars(args)
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=config,
            name=args.folder,
            save_code=True,
            group=args.wandb_group,
            tags=["td3", "separate_agents"]
        )
    learner_log_dir = f"{args.root}/{args.folder}/learner/"
    os.makedirs(learner_log_dir, exist_ok=True)
    writer = SummaryWriter(learner_log_dir)
    
    with open(os.path.join(learner_log_dir, "config.yaml"), "w") as f:
        yaml.dump(config_data, f)
        
    logger = Logger(log_wandb=args.track, tensorboard=writer, log_dir=learner_log_dir)

    # Agents
    agent_left = Td3Agent(args.obs_dim_left, args.action_dim // 2, args.action_high, args.policy_noise, device, args)
    agent_right = Td3Agent(args.obs_dim_right, args.action_dim // 2, args.action_high, args.policy_noise, device, args)

    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint)
        if 'agent_left' in ckpt:
            agent_left.load_state_dict(ckpt['agent_left'])
            agent_right.load_state_dict(ckpt['agent_right'])
        else:
            print("Checkpoint format mismatch? Expected keys 'agent_left' and 'agent_right'.")

    global_update = 0
    cumulative_times = defaultdict(float)
    while rb_left.n_inserts < args.learning_starts:
        if training_status['paused']:
            print("Learner paused (Actor shutdown)... (batch count: {})".format(rb_left.n_inserts), end='\r')
            time.sleep(1)
            continue
        time.sleep(2)

    while global_update < args.total_grad_updates:
        if training_status['paused']:
            print(f"Learner paused (Actor shutdown)... (update: {global_update})", end='\r')
            time.sleep(1)
            continue

        if rb_left.n_inserts * args.utd * args.chunk_size < global_update:
            print(f"Waiting for data... inserts: {rb_left.n_inserts}")
            time.sleep(1)
            continue
    
        update_time = time.perf_counter()
        global_update += 1
        

        # Sample
        data_left = next(replay_iterator_left)
        data_right = next(replay_iterator_right)

        # Update
        metrics_left = agent_left.update(data_left, global_update, rb_left.n_inserts, logger, prefix="left/")
        metrics_right = agent_right.update(data_right, global_update, rb_right.n_inserts, logger, prefix="right/")

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # Logging
        if global_update % args.log_freq == 0:
            logger.add_scalar("time/update_time", update_time, global_update)
            logger.add_scalar("stats/n_inserts", rb_left.n_inserts, global_update)
            
            if hasattr(rb_left, 'pos_buffer'):
                logger.add_scalar("stats/rb_left_pos_size", len(rb_left.pos_buffer), global_update)
                logger.add_scalar("stats/rb_left_neg_size", len(rb_left.neg_buffer), global_update)
            if hasattr(rb_right, 'pos_buffer'):
                logger.add_scalar("stats/rb_right_pos_size", len(rb_right.pos_buffer), global_update)
                logger.add_scalar("stats/rb_right_neg_size", len(rb_right.neg_buffer), global_update)

            if "recent_size" in sample_args_left:
                logger.add_scalar("stats/recent_size", sample_args_left["recent_size"], global_update)

            
            for k, v in metrics_left.items():
                logger.add_scalar(k, v, global_update)
            for k, v in metrics_right.items():
                logger.add_scalar(k, v, global_update)
                
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_update)
        
        # Saving
        if args.save_model and global_update % args.save_freq == 0:
            model_path = f"{args.root}/{args.folder}/ckpt_{global_update}.pt"
            save_dict = {
                'agent_left': agent_left.state_dict(),
                'agent_right': agent_right.state_dict(),
                'actor_left': agent_left.actor_target.state_dict(),
                'actor_right': agent_right.actor_target.state_dict(),
            }
            torch.save(save_dict, model_path)
            print(f"Model saved to {model_path}")

    # Final Save
    if args.save_model:
        model_path = f"{args.root}/{args.folder}/final_ckpt.pt"
        save_dict = {
            'agent_left': agent_left.state_dict(),
            'agent_right': agent_right.state_dict(),
             'actor_left': agent_left.actor_target.state_dict(),
             'actor_right': agent_right.actor_target.state_dict(),
        }
        torch.save(save_dict, model_path)
        print(f"Final model saved to {model_path}")
        writer.close()

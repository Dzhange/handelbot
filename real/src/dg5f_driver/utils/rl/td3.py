import csv
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import wandb


class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None, log_dir: str = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
        self.csv_file = None
        self.csv_writer = None
        
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self.csv_path = os.path.join(log_dir, "log.csv")
            self.csv_file = open(self.csv_path, "w", newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['step', 'tag', 'value'])

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        if self.writer:
            self.writer.add_scalar(tag, scalar_value, step)
        if self.csv_writer:
            self.csv_writer.writerow([step, tag, scalar_value])
            self.csv_file.flush()
            
    def close(self):
        if self.writer:
            self.writer.close()
        if self.csv_file:
            self.csv_file.close()

class QNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, critic_hidden_dims=[256, 256, 256], dropout=0.0, layernorm=False, history_length=1):
        super().__init__()
        
        layers = []
        input_dim = obs_dim * history_length + action_dim
        
        for h_dim in critic_hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            if layernorm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = h_dim
            
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Tanh())
        
        self.net = nn.Sequential(*layers)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return (self.net(x) + 1) * 6 # Manually set based on known rewards

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_high, policy_noise, actor_hidden_dims=[256, 256, 256], dropout=0.0, layernorm=False, history_length=1, noise_beta=0.9):
        super().__init__()
        
        layers = []
        input_dim = obs_dim * history_length
        
        for h_dim in actor_hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            if layernorm:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = h_dim
            
        self.backbone = nn.Sequential(*layers)
        self.fc = nn.Linear(input_dim, action_dim)
        
        # action rescaling
        self.action_scale_cpu = torch.tensor(action_high, dtype=torch.float32)
        self.action_bias_cpu = torch.tensor(0.0, dtype=torch.float32)
        self.register_buffer("action_scale", torch.tensor(action_high, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor(0.0, dtype=torch.float32))
        self.policy_noise = policy_noise
        self.noise_beta = noise_beta
        self.noise = None

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        action = torch.tanh(x) * self.action_scale + self.action_bias
        return action

    def get_eval_action(self, x):
        was_training = self.training
        self.eval()
        out = self.forward(x)
        if was_training:
            self.train()
        return out
        
    def get_action(self, x, policy_noise=None):
        action = self.forward(x)
        if policy_noise is None:
            policy_noise = self.policy_noise
        policy_noise = torch.tensor(policy_noise)
        
        if self.noise is None or self.noise.shape != action.shape:
             self.noise = torch.randn(action.size(), device='cpu')

        self.noise = self.noise_beta * self.noise + np.sqrt(1 - self.noise_beta**2) * torch.randn(action.size(), device='cpu')
        
        action = action.cpu()
        noise_scaled = (self.noise * policy_noise).clamp(-policy_noise * 5, policy_noise * 5) # Using 0.5 as clip for now, unrelated to noise_clip in learner
        return (action + noise_scaled).clamp(-self.action_scale_cpu, self.action_scale_cpu), None, None

    def get_guided_action(self, x, guidance, policy_noise=None):
        action = self.forward(x)
        if policy_noise is None:
            policy_noise = self.policy_noise
        policy_noise = torch.tensor(policy_noise)
        
        if self.noise is None or self.noise.shape != action.shape:
             self.noise = torch.randn(action.size(), device='cpu')
        self.noise = self.noise_beta * self.noise + np.sqrt(1 - self.noise_beta**2) * torch.randn(action.size(), device='cpu')
        
        # Guided Noise
        current_noise = self.noise.clone()
        mask = (np.expand_dims(guidance, axis=0) != 0)
        if mask.any():
            current_noise[mask] = torch.abs(current_noise[mask]) * guidance.unsqueeze(0)[mask].float()
        self.noise = current_noise
        
        action = action.cpu()
        noise_scaled = (self.noise * policy_noise).clamp(-policy_noise * 5, policy_noise * 5)
        return (action + noise_scaled).clamp(-self.action_scale_cpu, self.action_scale_cpu), None, None

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)

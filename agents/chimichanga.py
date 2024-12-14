import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from .drqv2 import Encoder, Actor, Critic

class Chimichanga(nn.Module):
    def __init__(self, repr_dim, feature_dim, action_shape, hidden_dim, encoder, device):
        super().__init__()
        self.encoder = encoder
        self.device = device
        
        # Project concatenated (s,a,s,a) sequence
        self.proj_sasa = nn.Sequential(
            nn.Linear(repr_dim*2 + action_shape[0]*2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        # For reward prediction
        self.reward = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        self.proj_s = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim), 
            nn.Tanh()
        )
        
        self.W = nn.Parameter(torch.rand(feature_dim, feature_dim))
        self.apply(utils.weight_init)

    def encode(self, x, ema=False):
        if ema:
            with torch.no_grad():
                z_out = self.proj_s(self.encoder(x))
        else:
            z_out = self.proj_s(self.encoder(x))
        return z_out

    def encode_sequence(self, states, actions):
        # Encode states
        s1, s2 = states[:,0], states[:,1]
        s1_enc = self.encoder(s1)
        s2_enc = self.encoder(s2)
        
        # Take corresponding actions
        a1, a2 = actions[:,0], actions[:,1]
        
        # Concatenate everything
        sasa = torch.cat([s1_enc, a1, s2_enc, a2], dim=-1)
        return self.proj_sasa(sasa)

    def compute_logits(self, z_a, z_pos):
        Wz = torch.matmul(self.W, z_pos.T)
        logits = torch.matmul(z_a, Wz)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

class ChimichangaAgent:
    def __init__(self, obs_shape, action_shape, device, lr, encoder_lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 reward, curl):
        
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        
        # Same core components as TACO/DrQV2
        self.encoder = Encoder(obs_shape, feature_dim).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                            hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                   feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.chimichanga = Chimichanga(
            self.encoder.repr_dim,
            feature_dim,
            action_shape,
            hidden_dim,
            self.encoder,
            device
        ).to(device)
        
        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.chimichanga_opt = torch.optim.Adam(self.chimichanga.parameters(), lr=encoder_lr)
        
        self.reward = reward
        self.curl = curl
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.train()
        
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update_chimichanga(self, obs, action, next_obs, reward):
        metrics = dict()
        
        # CURL loss
        if self.curl:
            obs_anchor = self.aug(obs.float())
            obs_pos = self.aug(obs.float())
            z_a = self.chimichanga.encode(obs_anchor)
            z_pos = self.chimichanga.encode(obs_pos, ema=True)
            logits = self.chimichanga.compute_logits(z_a, z_pos)
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            curl_loss = self.cross_entropy_loss(logits, labels)
        else:
            curl_loss = torch.tensor(0.)
            
        # Chimichanga contrastive loss
        z_seq = self.chimichanga.encode_sequence(obs, action)
        next_z = self.chimichanga.encode(self.aug(next_obs.float()), ema=True)
        logits = self.chimichanga.compute_logits(z_seq, next_z)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        chimichanga_loss = self.cross_entropy_loss(logits, labels)
        
        # Reward prediction loss
        if self.reward:
            reward_pred = self.chimichanga.reward(z_seq)
            reward_loss = F.mse_loss(reward_pred, reward)
        else:
            reward_loss = torch.tensor(0.)
        
        # Total loss
        total_loss = chimichanga_loss + curl_loss + reward_loss
        
        # Optimize
        self.encoder_opt.zero_grad(set_to_none=True)
        self.chimichanga_opt.zero_grad(set_to_none=True)
        total_loss.backward()
        self.encoder_opt.step()
        self.chimichanga_opt.step()
        
        if self.use_tb:
            metrics['reward_loss'] = reward_loss.item()
            metrics['curl_loss'] = curl_loss.item()
            metrics['chimichanga_loss'] = chimichanga_loss.item()
        
        return metrics

    def update(self, replay_iter, step):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics
        
        batch = next(replay_iter)
        obs, action, states_actions_seq, reward, discount, next_obs, r_next_obs = utils.to_torch(
            batch, self.device)

        # Regular DrQv2 updates
        obs_en = self.aug(obs.float())
        next_obs_en = self.aug(next_obs.float())
        obs_en = self.encoder(obs_en)
        with torch.no_grad():
            next_obs_en = self.encoder(next_obs_en)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        metrics.update(
            self.update_critic(obs_en, action, reward, discount, next_obs_en, step))
        metrics.update(self.update_actor(obs_en.detach(), step))
        metrics.update(
            self.update_chimichanga(states_actions_seq[0], states_actions_seq[1], r_next_obs, reward))

        utils.soft_update_params(self.critic, self.critic_target,
                               self.critic_target_tau)

        return metrics
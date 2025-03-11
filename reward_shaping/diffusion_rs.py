"""
Diffusion-based reward shaping module that can be integrated with any DRL algorithm.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym



class DRESS:
    """
    A modular diffusion-based reward shaping component that can be integrated with any DRL algorithm.
    """

    def __init__(
        self,
        env,
        actor_class,
        critic_class,  # Added critic class parameter
        model,
        buffer_size=1000000,
        batch_size=256,
        learning_starts=500,
        reward_scale=1.0,
        beta=0.2,
        device="cpu",
        seed=1,
        alpha=0.2,
        alpha_autotune=True,
        critic_lr=1e-4,
        actor_lr=3e-4,
        alpha_lr=1e-4,
        tau=0.005,
    ):
        self.env = env
        self.device = torch.device(device)
        self.reward_scale = reward_scale
        self.beta = beta
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau

        # Create reward space
        self.reward_space = gym.spaces.Box(
            low=-reward_scale,
            high=reward_scale,
            shape=(1,),
            dtype=np.float32
        )

        # Create state-action observation space
        self.obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(env.observation_space.shape[0] + env.action_space.shape[0],),
            dtype=np.float32
        )

        # Initialize networks
        self.actor = actor_class(
            self.obs_space,
            self.reward_space,
            model
        ).to(device)

        # Initialize critic networks (twin critics like in original code)
        self.critic1 = critic_class(self.obs_space, self.reward_space).to(device)
        self.critic2 = critic_class(self.obs_space, self.reward_space).to(device)
        self.critic1_target = critic_class(self.obs_space, self.reward_space).to(device)
        self.critic2_target = critic_class(self.obs_space, self.reward_space).to(device)

        # Copy weights to target networks
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr
        )

        # Initialize alpha (temperature parameter)
        self.alpha_autotune = alpha_autotune
        if alpha_autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.reward_space.shape)).to(device)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = alpha

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            self.obs_space,
            self.reward_space,
            device=device,
            handle_timeout_termination=False
        )

        self.total_steps = 0

    def get_reward(self, obs_ra):
        """Get shaped reward for a state-action pair."""
        if self.total_steps < self.learning_starts:
            return 0.0
        # CPU
        with torch.no_grad():
            shaped_reward, _, _ = self.actor.get_action(obs_ra)
            shaped_reward = shaped_reward.detach().cpu().numpy()[0]
            return self.beta * shaped_reward
        # GPU
        # with torch.no_grad():
        #     obs_ra = torch.as_tensor(obs_ra).to(self.device)
        #     shaped_reward, _, _ = self.actor.get_action(obs_ra)
        #     shaped_reward = shaped_reward.detach()[0]
        #     return self.beta * shaped_reward


    def store_transition(self, obs, action, next_obs, next_action, env_reward, done):
        """Store transition in replay buffer."""
        obs_action = np.concatenate([obs, action])
        next_obs_action = np.concatenate([next_obs, next_action])

        self.replay_buffer.add(
            obs_action,
            next_obs_action,
            self.reward_space.sample(),  # Initial reward proposal
            env_reward,
            done,
            {}
        )

        self.total_steps += 1

    def update(self, drl_batch):
        """Update the diffusion model using twin critics like in original code."""
        if self.total_steps < self.learning_starts:
            return {}

        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)

        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.get_action(batch.next_observations)
            q1_next = self.critic1_target(batch.next_observations, next_actions)
            q2_next = self.critic2_target(batch.next_observations, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            next_q_value = batch.rewards.flatten() + \
                          (1 - batch.dones.flatten()) * 0.99 * min_q_next.view(-1)

        # Current Q values
        q1 = self.critic1(batch.observations, batch.actions).view(-1)
        q2 = self.critic2(batch.observations, batch.actions).view(-1)

        # Compute critic loss
        critic1_loss = F.mse_loss(q1, next_q_value)
        critic2_loss = F.mse_loss(q2, next_q_value)
        critic_loss = critic1_loss + critic2_loss

        # Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actions, log_probs, _ = self.actor.get_action(batch.observations)
        q1_pi = self.critic1(batch.observations, actions)
        q2_pi = self.critic2(batch.observations, actions)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_probs - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha if needed
        if self.alpha_autotune:
            with torch.no_grad():
                _, log_probs, _ = self.actor.get_action(batch.observations)
            alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

        # Update target networks
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'RSreward_shaping_loss': actor_loss.item(),
            'RScritic1_loss': critic1_loss.item(),
            'RScritic2_loss': critic2_loss.item(),
            'RSalpha': self.alpha
        }

    def save(self, path):
        """Save the reward shaping model."""
        torch.save(
            {
                'actor_state_dict': self.actor.state_dict(),
                'critic1_state_dict': self.critic1.state_dict(),
                'critic2_state_dict': self.critic2.state_dict(),
                'critic1_target_state_dict': self.critic1_target.state_dict(),
                'critic2_target_state_dict': self.critic2_target.state_dict(),
                'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
                'alpha': self.alpha,
                'total_steps': self.total_steps
            },
            path
        )

    def load(self, path):
        """Load the reward shaping model."""
        checkpoint = torch.load(path, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.alpha = checkpoint['alpha']
        self.total_steps = checkpoint['total_steps']
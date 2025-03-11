import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.buffers import ReplayBuffer
import os
import datetime
import time
import wandb


class SAC:
    def __init__(
            self,
            env,
            actor_class,
            critic_class,
            reward_shaping,
            exp_name="SAC",
            seed=1,
            buffer_size=int(1e6),
            batch_size=256,
            actor_lr=3e-4,
            critic_lr=1e-3,
            alpha_lr=1e-4,
            tau=0.005,
            gamma=0.99,
            alpha=0.2,
            alpha_autotune=True,
            device="cuda:0",  
            save_folder="./sac/",
            write_frequency=100,
    ):
        self.device = torch.device(device)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  
        np.random.seed(seed)

        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.reward_shaping = reward_shaping
        self.exp_name = exp_name
        self.seed = seed
        self.alpha = alpha
        self.alpha_autotune = alpha_autotune
        self.actor = actor_class(env.observation_space, env.action_space).to(self.device)
        self.qf_1 = critic_class(env.observation_space, env.action_space).to(self.device)
        self.qf_2 = critic_class(env.observation_space, env.action_space).to(self.device)
        self.qf_1_target = critic_class(env.observation_space, env.action_space).to(self.device)
        self.qf_2_target = critic_class(env.observation_space, env.action_space).to(self.device)

        self.qf_1_target.load_state_dict(self.qf_1.state_dict())
        self.qf_2_target.load_state_dict(self.qf_2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            list(self.qf_1.parameters()) + list(self.qf_2.parameters()),
            lr=critic_lr
        )

        if alpha_autotune:
            self.target_entropy = -np.prod(env.action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(
            buffer_size,
            env.observation_space,
            env.action_space,
            device=self.device,
            handle_timeout_termination=False
        )

        run_name = f"{exp_name}-{env.unwrapped.spec.id}-{seed}-{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')}"
        os.makedirs("./runs/", exist_ok=True)
        self.writer = SummaryWriter(os.path.join("./runs/", run_name))
        self.write_frequency = write_frequency
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)

    def optimize(self, global_step):
        """Optimize actor and critic networks."""
        data = self.replay_buffer.sample(self.batch_size)

        # Update critic
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor.get_action(data.next_observations)
            qf_1_next = self.qf_1_target(data.next_observations, next_actions)
            qf_2_next = self.qf_2_target(data.next_observations, next_actions)
            min_qf_next = torch.min(qf_1_next, qf_2_next) - self.alpha * next_log_pi

            if self.reward_shaping:
                obs_ra = torch.cat((data.observations, data.actions), dim=1)
                shaped_reward = self.reward_shaping.get_reward(obs_ra)
                # print('data.rewards.flatten()', data.rewards.flatten())
                next_q_value = data.rewards.flatten() +  shaped_reward + \
                              (1 - data.dones.flatten()) * self.gamma * min_qf_next.view(-1)
            else:
                next_q_value = data.rewards.flatten() + \
                              (1 - data.dones.flatten()) * self.gamma * min_qf_next.view(-1)

        # Get current Q values
        q1_pred = self.qf_1(data.observations, data.actions).view(-1)
        q2_pred = self.qf_2(data.observations, data.actions).view(-1)

        # Compute critic loss
        qf_1_loss = F.mse_loss(q1_pred, next_q_value)
        qf_2_loss = F.mse_loss(q2_pred, next_q_value)
        qf_loss = qf_1_loss + qf_2_loss

        # Optimize critics
        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actions_pi, log_pi, _ = self.actor.get_action(data.observations)
        qf_1_pi = self.qf_1(data.observations, actions_pi)
        qf_2_pi = self.qf_2(data.observations, actions_pi)
        min_qf_pi = torch.min(qf_1_pi, qf_2_pi)
        actor_loss = (self.alpha * log_pi - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update alpha if auto-tuning
        if self.alpha_autotune:
            with torch.no_grad():
                _, log_pi, _ = self.actor.get_action(data.observations)
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # Update target networks
        for param, target_param in zip(self.qf_1.parameters(), self.qf_1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.qf_2.parameters(), self.qf_2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Log metrics
        if global_step % self.write_frequency == 0:
            # self.writer.add_scalar("losses/q1_values", q1_pred.mean().item(), global_step)
            # self.writer.add_scalar("losses/q2_values", q2_pred.mean().item(), global_step)
            # self.writer.add_scalar("losses/q1_loss", qf_1_loss.item(), global_step)
            # self.writer.add_scalar("losses/q2_loss", qf_2_loss.item(), global_step)
            # self.writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
            # self.writer.add_scalar("losses/alpha", self.alpha, global_step)
            wandb.log({
                "losses/q1_values": q1_pred.mean().item(),
                "losses/q2_values": q2_pred.mean().item(),
                "losses/q1_loss": qf_1_loss.item(),
                "losses/q2_loss": qf_2_loss.item(),
                "losses/actor_loss": actor_loss.item(),
                "losses/alpha": self.alpha,
            }, step=global_step)

            if self.alpha_autotune:
                wandb.log({"losses/alpha_loss": alpha_loss.item()}, step=global_step)
                # self.writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    def learn(self, total_timesteps, learning_starts=1000, save_frequency=10000):
        """Main training loop that trains both SAC (PA) and DRESS (RA)."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_rs_reward = 0
        episode_steps = 0

        for global_step in range(total_timesteps):
            if global_step < learning_starts:
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    action, _, _ = self.actor.get_action(torch.FloatTensor(obs).to(self.device).unsqueeze(0))
                    action = action.cpu().numpy()[0]

            next_obs, reward, done, term, info = self.env.step(action)

            if global_step < learning_starts:
                next_action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    next_action, _, _ = self.actor.get_action(torch.FloatTensor(next_obs).to(self.device).unsqueeze(0))
                    next_action = next_action.cpu().numpy()[0]

            self.replay_buffer.add(obs, next_obs, action, reward, done, info)

            if self.reward_shaping:
                obs_np = obs if isinstance(obs, np.ndarray) else obs.cpu().numpy()
                action_np = action if isinstance(action, np.ndarray) else action.cpu().numpy()
                next_obs_np = next_obs if isinstance(next_obs, np.ndarray) else next_obs.cpu().numpy()
                next_action_np = next_action if isinstance(next_action, np.ndarray) else next_action.cpu().numpy()

                self.reward_shaping.store_transition(obs_np, action_np, next_obs_np, next_action_np, reward, done)

                obs_action = np.hstack((obs_np, action_np))
                shaped_reward = self.reward_shaping.get_reward(
                    torch.FloatTensor(np.expand_dims(obs_action, axis=0)).to(self.device)
                )

                if global_step >= learning_starts:
                    rs_info = self.reward_shaping.update(None)
                    wandb.log(rs_info, step=global_step)

                episode_rs_reward += shaped_reward

            if global_step >= learning_starts:
                self.optimize(global_step)

            obs = next_obs if not (done or term) else self.env.reset()[0]
            episode_reward += reward
            episode_steps += 1

            if done or term:
                print(f"global_step={global_step}, episodic_return={episode_reward}, rs_return={episode_rs_reward}")
                wandb.log({
                    "charts/episodic_return": episode_reward,
                    "charts/rs_episodic_return": episode_rs_reward,
                    "charts/episodic_length": episode_steps
                }, step=global_step)

                episode_reward = 0
                episode_rs_reward = 0
                episode_steps = 0

            if (global_step + 1) % save_frequency == 0:
                self.save(f"step_{global_step}")
                if self.reward_shaping:
                    self.reward_shaping.save(os.path.join(self.save_folder, f"dress_step_{global_step}.pt"))

        self.env.close()
        self.writer.close()

    def save(self, tag):
        """Save model checkpoint."""
        save_path = os.path.join(self.save_folder, f"sac_{tag}")
        os.makedirs(save_path, exist_ok=True)

        # Save model state
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'qf1_state_dict': self.qf_1.state_dict(),
            'qf2_state_dict': self.qf_2.state_dict(),
            'qf1_target_state_dict': self.qf_1_target.state_dict(),
            'qf2_target_state_dict': self.qf_2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'alpha': self.alpha
        }

        if self.alpha_autotune:
            checkpoint.update({
                'log_alpha': self.log_alpha,
                'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict()
            })

        torch.save(checkpoint, os.path.join(save_path, f"{self.exp_name}-{tag}-{self.seed}.pt"))

"""
Main script to run DRESSed-DRL algorithms
"""

import argparse
import gymnasium as gym
from envs import mec_environment
from reward_shaping.diffusion_rs import DRESS
from drl_algorithms.DRL_SAC import SAC
from reward_shaping.utils import robotics_env_maker, classic_control_env_maker
from reward_shaping.networks import (BasicActorSAC, ActorResidual, BasicQNetwork, QNetworkResidual, Diffusion, MLP)
import numpy as np
import torch
import os
import wandb


def parse_args():
    parser = argparse.ArgumentParser()

    # Environment settings
    parser.add_argument("--env-id", type=str, default="BipedalWalker-v3", help="Environment ID")
    # parser.add_argument("--env_id", type=str, default="MEC-v1", help="Environment ID")
    parser.add_argument('--algorithm', type=str, default='sac', help='DRL algorithm to use')
    parser.add_argument("--render", type=bool, default=False)

    # Experiment settings
    # parser.add_argument("--exp_name", type=str, default="DRL-SAC")
    parser.add_argument("--exp-name", type=str, default="DressedDRL")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cuda", type=int, default=0)  # GPU
    # parser.add_argument("--cuda", type=int, default=-1)  # Use CPU by default

    # SAC parameters
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--alpha-lr", type=float, default=1e-4)

    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--alpha-autotune", type=bool, default=True)

    ########## DRESS parameters
    parser.add_argument("--s-reward-shaping", type=bool, default=True)
    parser.add_argument("--use-diffusion", type=bool, default=True)
    parser.add_argument("--beta", type=float, default=0.2)

    parser.add_argument("--rs-buffer-size", type=int, default=1000000)
    parser.add_argument("--rs-batch-size", type=int, default=256)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--rs-learning-starts", type=int, default=5e3)
    # parser.add_argument("--rs-learning-starts", type=int, default=1000)

    # Training parameters
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--learning-starts", type=int, default=1e4)
    # parser.add_argument("--learning-starts", type=int, default=1000)

    # Logging parameters
    parser.add_argument("--log_dir", type=str, default="runs", help="Directory to save logs")
    parser.add_argument("--write-frequency", type=int, default=100)
    parser.add_argument("--save-frequency", type=int, default=10000)
    parser.add_argument("--plot-rewards", action="store_true", help="Plot rewards after training")

    return parser.parse_args()


def main():
    args = parse_args()
    wandb.init(
        project="DRESS",
        config=vars(args),
        name=f"{args.algorithm}-{args.exp_name}-{args.env_id}-{args.seed}"
    )

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    # env = gym.make(args.env_id)
    env = robotics_env_maker(env_id=args.env_id, seed=args.seed, render=args.render) if args.env_id.startswith(
        "My") else classic_control_env_maker(env_id=args.env_id, seed=args.seed, render=args.render)

    # Setup device
    device = f"cuda:{args.cuda}" if args.cuda >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    log_dir = os.path.join(args.log_dir, f"{args.exp_name}-{args.env_id}-{args.seed}")
    os.makedirs(log_dir, exist_ok=True)


    if args.s_reward_shaping:
        # Initialize DRESS
        actor_class = Diffusion if args.use_diffusion else ActorResidual
        reward_shaping = DRESS(
            env=env,
            actor_class=actor_class,
            # ActorResidual
            critic_class=QNetworkResidual,
            model=MLP(
                state_dim=env.observation_space.shape[0] + env.action_space.shape[0],
                action_dim=128
            ),
            # model=MLP,
            buffer_size=args.rs_buffer_size,
            batch_size=args.rs_batch_size,
            learning_starts=args.rs_learning_starts,
            reward_scale=args.reward_scale,
            beta=args.beta,
            device=device,
            seed=args.seed
        )
    else:
        reward_shaping = None

    def init_drl_model(algorithm_name, env, args, device, log_dir, reward_shaping):
        algorithms = {
            'sac': lambda: SAC(
                env=env,
                actor_class=BasicActorSAC,
                critic_class=BasicQNetwork,
                reward_shaping=reward_shaping,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                alpha_lr=args.alpha_lr,
                tau=args.tau,
                gamma=args.gamma,
                alpha=args.alpha,
                alpha_autotune=args.alpha_autotune,
                device=device,
                seed=args.seed,
                save_folder=log_dir,
                write_frequency=args.write_frequency
            )
        }

        if algorithm_name.lower() not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}. Available options: {list(algorithms.keys())}")

        drlmodel = algorithms[algorithm_name.lower()]()

        return drlmodel

    # Usage in main:
    algorithm_name = args.algorithm  # Add this to your argument parser
    drlmodel = init_drl_model(algorithm_name, env, args, device, log_dir, reward_shaping)

    print("Starting training...")
    drlmodel.learn(
        total_timesteps=args.total_timesteps,
        learning_starts=args.learning_starts,
        save_frequency=args.save_frequency
    )
    drlmodel.save("final")
    print(f"Training finished. Logs and models are saved in {log_dir}")

if __name__ == "__main__":
    main()

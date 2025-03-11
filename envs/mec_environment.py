import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register
import torch.nn.functional as F
import torch

class MECEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self, num_users=10, max_bandwidth=20, max_power=10, max_computation=100, latency_threshold=0.03, max_steps=10
    ):
        super(MECEnvironment, self).__init__()
        self.num_users = num_users
        self.max_bandwidth = max_bandwidth
        self.max_power = max_power
        self.max_computation = max_computation
        self.latency_threshold = latency_threshold
        self.max_steps = max_steps

        # Observation space: [channel_quality, user_demand, latency_requirement, user_position]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_users * 4,), dtype=np.float32
        )

        # Action space: [bandwidth_allocation, power_allocation, computation_allocation]
        self.action_space = spaces.Box(
            low=0, high=1, shape=(num_users * 3,), dtype=np.float32
        )

        self.channel_quality = None
        self.user_demand = None
        self.latency_requirement = None
        self.user_positions = None
        self.step_count = 0
        self.user_patience = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        
        # ================== Reset your state here ================== ==============
        self.channel_quality = np.random.uniform(0.1, 1, size=self.num_users).astype(np.float32)
        self.user_demand = np.random.uniform(0.5, 1, size=self.num_users).astype(np.float32)
        self.latency_requirement = np.random.uniform(0.1, 1, size=self.num_users).astype(np.float32)
        self.user_positions = np.random.uniform(0, 1, size=self.num_users).astype(np.float32)
        self.user_patience = np.random.uniform(5, 15, size=self.num_users).astype(np.float32)  # Time before leaving

        self.step_count = 0
        state = np.concatenate([
            self.channel_quality, self.user_demand, self.latency_requirement, self.user_positions
        ])
        # ================== ================== ================== =================
        
        return state, {}

    def step(self, action):
        # Ensure action

        # ================== Define your reward here ================== ============
        reward = np.random.uniform(0.1, 1)
        # ================== ================== ================== =================
        
        # ================== Update your state here ================== =============
        self.user_positions += np.random.uniform(-0.05, 0.05, size=self.num_users)
        self.user_positions = np.clip(self.user_positions, 0, 1)
        self.channel_quality = np.random.uniform(0.1, 1, size=self.num_users).astype(np.float64)
        self.user_demand = np.random.uniform(0.1, 1, size=self.num_users).astype(np.float64)
        self.latency_requirement = np.random.uniform(0.1, 1, size=self.num_users).astype(np.float64)
        next_state = np.concatenate([
            self.channel_quality, self.user_demand, self.latency_requirement, self.user_positions
        ])
        # ================== ================== ================== =================
        
        # ================== Update your done mark ================== ==============
        self.step_count += 1
        done = self.step_count >= self.max_steps
        # ================== ================== ================== =================

        info = {
            "reward": reward,
        }

        return next_state, reward, done, False, info

# Register MEC environment
register(
    id="MEC-v1",
    entry_point="envs.mec_environment:MECEnvironment"
    # kwargs={"num_users": 5, "max_bandwidth": 20, "max_power": 10, "max_computation": 100},
)

if __name__ == "__main__":
    env = MECEnvironment()
    state, _ = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        state, reward, done, _, info = env.step(action)
    print("Episode finished.", info)

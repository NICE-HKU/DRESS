import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .helpers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract,
    Losses
)
from .utils import Progress, Silent
from .helpers import SinusoidalPosEmb

class BasicActorSAC(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(action_space.shape))
        # action rescaling
        self.register_buffer("action_scale",
                             torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",
                             torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))
        self.log_std_max = 2
        self.log_std_min = -5

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU()):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(out_features, out_features)
        self.activation = activation

    def forward(self, x):
        residual = x
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x += residual
        x = self.activation(x)
        return x


class QNetworkResidual(nn.Module):
    def __init__(self, observation_space, action_space, block_num=3):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod() + np.prod(action_space.shape), 256)
        self.hidden_blocks = nn.ModuleList([ResidualBlock(256, 256) for _ in range(block_num)])
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.fc1(x)
        for block in self.hidden_blocks:
            x = block(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorResidual(nn.Module):

    def __init__(self, observation_space, action_space, MLP, block_num=3):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod(), 256)
        self.hidden_blocks = nn.ModuleList([ResidualBlock(256, 256) for _ in range(block_num)])
        self.fc2 = nn.Linear(256, 128)
        self.fc_mean = nn.Linear(128, np.prod(action_space.shape))
        self.fc_logstd = nn.Linear(128, np.prod(action_space.shape))
        # action rescaling
        self.register_buffer("action_scale",
                             torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",
                             torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32))
        self.log_std_max = 2
        self.log_std_min = -5

    def forward(self, x):
        x = self.fc1(x)
        for block in self.hidden_blocks:
            x = block(x)
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

class MLP(nn.Module):
    def __init__(
        self,
        state_dim = 10,
        action_dim = 10,
        hidden_dim=256,
        t_dim=16,
        activation='mish'
    ):
        super(MLP, self).__init__()
        _act = nn.Mish if activation == 'mish' else nn.ReLU
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            _act(),
            nn.Linear(t_dim * 2, t_dim),
        )
        self.mid_layer = nn.Sequential(
            nn.Linear(hidden_dim + action_dim + t_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x, time, state):
        processed_state = self.state_mlp(state)
        t = self.time_mlp(time)
        x = torch.cat([x, t, processed_state], dim=1)
        x = self.mid_layer(x)
        return x

class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, model, max_action = 1,
                 beta_schedule='vp', n_timesteps=5,
                 loss_type='l2', clip_denoised=True):
        # Call parent constructor
        super(Diffusion, self).__init__()

        # Set initial attributes
        self.state_dim = np.array(state_dim.shape).prod()
        self.action_dim = 128
        self.max_action = max_action
        self.model = model
        self.model.state_dim = np.array(state_dim.shape).prod()
        self.model.action_dim = np.array(action_dim.shape).prod()
        # action rescaling
        self.register_buffer("action_scale",
                             torch.tensor((action_dim.high - action_dim.low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",
                             torch.tensor((action_dim.high + action_dim.low) / 2.0, dtype=torch.float32))
        self.log_std_max = 2
        self.log_std_min = -5
        self.fc_mean = nn.Linear(128, np.prod(action_dim.shape))
        self.fc_logstd = nn.Linear(128, np.prod(action_dim.shape))

        # Define the diffusion beta schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        # Define alpha parameters related to the beta schedule
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised

        # Register these values as buffers in the module, which PyTorch will track
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Pre-calculate some quantities for the diffusion process and posterior
        # distribution calculation based on alpha and beta schedules
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # More pre-calculations for the posterior distribution
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # Log calculation clipped to avoid log(0)
        # ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        # Select the appropriate loss function from the predefined Losses dictionary
        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#
    # Section to define the sampling methods for the diffusion
    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device

        # 1
        model_output = self.model(x, t, s)
        x_recon = (
                extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x.shape) * model_output
        )

        # 2
        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            raise RuntimeError("clip_denoised is disabled")

        # 3
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x.shape) * x_recon +
                extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)

        # 4
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return posterior_mean + nonzero_mask * (0.5 * posterior_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)
            progress.update({'t': i})
            if return_diffusion: diffusion.append(x)

        progress.close()
        return x

    # @torch.no_grad()
    # Generate a sample by using the p_sample_loop method and clamp the values within the max action range
    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)

        action = self.p_sample_loop(state, shape, *args, **kwargs)

        return action.clamp_(-self.max_action, self.max_action)

    # Define the sampling method for the posterior distribution
    def q_sample(self, x_start, t, noise=None):
        # if noise is not provided, generate random noise
        if noise is None:
            noise = torch.randn_like(x_start)
        # compute the diffused state
        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    # Generate a sample from the model
    def get_action(self, state, *args, **kwargs):
        # print(state)
        x = self.sample(state, *args, **kwargs)
        x = F.relu(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean

import torch
import numpy as np
from typing import Tuple, List


def conjugate_gradient(mvp_function, b: torch.Tensor, nsteps: int, residual_tol: float = 1e-10) -> torch.Tensor:
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()

    r_norm_sq = r.dot(r)

    for i in range(nsteps):
        Ap = mvp_function(p)
        alpha = r_norm_sq / (p.dot(Ap) + 1e-8)

        x += alpha * p
        r -= alpha * Ap

        r_norm_sq_new = r.dot(r)
        beta = r_norm_sq_new / (r_norm_sq + 1e-8)
        r_norm_sq = r_norm_sq_new

        if r_norm_sq < residual_tol:
            break

        p = r + beta * p

    return x


def get_flat_params(model: torch.nn.Module) -> torch.Tensor:
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    return torch.cat(params)


def set_flat_params(model: torch.nn.Module, flat_params: torch.Tensor):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad(model: torch.nn.Module, grad_grad: bool = False) -> torch.Tensor:
    grads = []
    for param in model.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            if param.grad is None:
                continue
            grads.append(param.grad.view(-1))
    return torch.cat(grads)


def compute_policy_gradient(policy, states: torch.Tensor, actions: torch.Tensor,
                            advantages: torch.Tensor) -> torch.Tensor:

    log_probs, _ = policy.evaluate_actions(states, actions)
    policy_gradient = -(advantages * log_probs).mean()
    return policy_gradient


def line_search(model, f, x: torch.Tensor, fullstep: torch.Tensor, expected_improve_rate: float,
                max_backtracks: int = 10, accept_ratio: float = .1) -> Tuple[bool, torch.Tensor]:

    fval = f().item()

    for stepfrac in .5 ** np.arange(max_backtracks):
        xnew = x + stepfrac * fullstep
        set_flat_params(model, xnew)
        newfval = f().item()

        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve

        if ratio > accept_ratio and actual_improve > 0:
            return True, xnew

    return False, x


def compute_kl(model, states: torch.Tensor, old_means: torch.Tensor,
               old_stds: torch.Tensor) -> torch.Tensor:

    new_means, new_stds = model(states)
    old_dist = torch.distributions.Normal(old_means, old_stds)
    new_dist = torch.distributions.Normal(new_means, new_stds)
    kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()
    return kl


def compute_losses(advantages: torch.Tensor, returns: torch.Tensor,
                   values: torch.Tensor, entropy: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    
    value_loss = (returns - values).pow(2).mean()
    entropy_loss = -entropy.mean()

    return value_loss, entropy_loss
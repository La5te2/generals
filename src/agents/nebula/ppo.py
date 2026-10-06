"""Terminal rewards train a clipped policy objective and a smoothed categorical value distribution."""

import math

import torch
from torch.nn import functional as F


def advantages(rewards, values, following, terminal, truncated, gamma, decay):
    result = torch.empty_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for step in range(rewards.shape[0] - 1, -1, -1):
        # a time limit bootstraps from the final board, while both kinds of boundary end the trace.
        bootstrap = following[step] * (~terminal[step])
        delta = rewards[step] + gamma * bootstrap - values[step]
        carry = delta + gamma * decay * (~(terminal[step] | truncated[step])) * carry
        result[step] = carry
    return result


def histogram(target, centers, sigma):
    target = target.clamp(centers[0], centers[-1]).unsqueeze(-1)
    half = (centers[1] - centers[0]) / 2
    upper = (centers + half - target) / (sigma * math.sqrt(2))
    lower = (centers - half - target) / (sigma * math.sqrt(2))
    weights = 0.5 * (upper.erf() - lower.erf())
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)


def objective(logits, critic, actions, previous, advantage, returns, centers, config, entropy):
    distribution = torch.distributions.Categorical(logits=logits)
    logprob = distribution.log_prob(actions)
    ratio = (logprob - previous).exp()
    policy = -torch.minimum(ratio * advantage, ratio.clamp(1 - config.clip, 1 + config.clip) * advantage).mean()
    target = histogram(returns, centers, config.sigma)
    value = -(target * F.log_softmax(critic, dim=-1)).sum(-1).mean()
    uncertainty = distribution.entropy().mean()
    loss = policy + config.value * value - entropy * uncertainty
    stats = torch.stack((policy.detach(), value.detach(), uncertainty.detach(),
                         (ratio - 1 - (logprob - previous)).mean().detach()))
    return loss, stats


@torch.no_grad()
def average(ema, current, decay):
    # one EMA update follows a complete training iteration, rather than each minibatch.
    for target, source in zip(ema.parameters(), current.parameters(), strict=True):
        target.lerp_(source, 1 - decay)

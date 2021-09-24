"""
    Code for defining MLPActorCritic
    Taken from: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py
"""
import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def mlp(sizes:List, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class Actor(nn.Module):
    def _distribution(self, obs, act_mask=None):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None, act_mask=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs, act_mask)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim:int, act_dim:int, hidden_sizes:Tuple, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs, act_mask=None):
        logits = self.logits_net(obs)
        if act_mask is not None:
            # multiply with mask so that invalid actions are zeroed out
            # changing probs after making distribution would mean that 
            # tensors would lose gradients attached to them and hence would 
            # not work, hence have to multiply with mask so that gradients 
            # are intact
            logits = logits * act_mask
        dist = Categorical(logits=logits)
        return dist

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class MLPGaussianActor(Actor):
    def __init__(self, obs_dim:int, act_dim:int, hidden_sizes:Tuple, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        # Last axis sum needed for Torch Normal distribution
        return pi.log_prob(act).sum(axis=-1)


class MLPCritic(nn.Module):
    def __init__(self, obs_dim:int, hidden_sizes:Tuple, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # Critical to ensure v has right shape.
        return torch.squeeze(self.v_net(obs), -1)

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, 
                hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], 
                                        hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, 
                                        hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs, act_mask=None, deterministic:bool=False):
        # return the action with the highest probability (Greedy); 
        # Use only for evaluation of policy
        if deterministic:
            with torch.no_grad():
                pi = self.pi._distribution(obs, act_mask)
                a = pi.probs.argmax().numpy()
            return a, 0, 0, pi.probs.numpy() #pi.logits.numpy()
        else:
            with torch.no_grad():
                pi = self.pi._distribution(obs, act_mask)
                a = pi.sample()
                logp_a = self.pi._log_prob_from_distribution(pi, a)
                v = self.v(obs)
            return a.numpy(), v.numpy(), logp_a.numpy(), pi.probs.numpy()

    def act(self, obs):
        return self.step(obs)[0]
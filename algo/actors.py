import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from HAD.utils.utils import hard_update


class Actor_Net(nn.Module):
    def __init__(self, state_dim, command_dim, out_dim, hidden_dim=128, norm_in=False):
        super(Actor_Net, self).__init__()
        if norm_in:
            self.in_fn = nn.BatchNorm1d(state_dim, affine=False)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(state_dim + command_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.nonlin = F.leaky_relu
        self.mean_linear = nn.Linear(hidden_dim, out_dim)
        self.std_linear = nn.Linear(hidden_dim, out_dim)
        self.a_bound = 2000
        self.std_max = 2
        self.std_min = -20
        self.action_min = 5
        self.action_max = 10

    def forward(self, state, command, return_log_pi=False, curr=0):
        inp = self.in_fn(state)
        inp = torch.cat((inp, command), dim=1)
        h1 = self.nonlin(self.fc1(inp))
        h2 = self.nonlin(self.fc2(h1))
        mean = self.mean_linear(h2)
        log_std = self.std_linear(h2)
        log_std = torch.clamp(log_std, self.std_min, self.std_max)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        tanh_action = torch.tanh(z)
        a_bound = self.env_curr(curr)
        action = a_bound * (tanh_action + 1)
        rets = [action]
        if return_log_pi:
            normal_log_prob = normal.log_prob(z)
            log_prob = normal_log_prob.sum(dim=1, keepdim=True)
            log_prob -= torch.log((1 - tanh_action.pow(2)) + 1e-6).sum(dim=1, keepdim=True)
            rets.append(log_prob)
        if len(rets) == 1:
            return rets[0]
        return rets

    def env_curr(self, curr):
        progress = min(1.0, curr / self.a_bound)
        bound = self.action_min + (self.action_max - self.action_min) * (1 - progress)
        return bound

class AttentionActor(object):
    def __init__(self, state_dim, command_dim, out_dim, hidden_dim=128, norm_in=False, lr=0.001):
        self.policy = Actor_Net(state_dim, command_dim, out_dim, hidden_dim, norm_in)
        self.target_policy = Actor_Net(state_dim, command_dim, out_dim, hidden_dim, norm_in)

        hard_update(self.target_policy, self.target_policy)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    # 功能类似于choose_action
    def step(self, state, command, return_log_pi=False, curr=0):
        return self.policy(state, command, return_log_pi, curr)



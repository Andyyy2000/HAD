import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from HAD.utils.utils import hard_update, soft_update
from HAD.algo.DuelingDQN import High_DDQN
from HAD.algo.actors import AttentionActor
from HAD.algo.critics import AttentionCritic

num_agents = 15
num_users = 8
MSELoss = nn.MSELoss()

def disable_gradients(module):
    for p in module.parameters():
        p.requires_grad = False

def enable_gradients(module):
    for p in module.parameters():
        p.requires_grad = True

class HAC(object):
    def __init__(self, sa_sizes, gamma=0.95, tau=1e-3, pi_lr=0.001, q_lr=0.001, high_lr=0.001, reward_scale=10.,
                 state_dim1=(15, 8), state_dim2=8, action_dim=(15, 8),                        # 高层网络参数
                 pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4, num_agents=15, batch_size=128, epsilon=0.05):          # 低层网络参数
        self.highlevel_net = High_DDQN(state_dim1, state_dim2, action_dim, epsilon=epsilon)
        self.highlevel_targetnet = High_DDQN(state_dim1, state_dim2, action_dim, epsilon=epsilon)

        self.nagents = len(sa_sizes)
        self.agents = [AttentionActor(state_dim=num_users, command_dim=num_users, out_dim=num_users, hidden_dim=pol_hidden_dim, norm_in=False) for _ in range(num_agents)]
        self.critic = AttentionCritic(sa_sizes, hidden_dim=critic_hidden_dim, attend_heads=attend_heads)
        self.target_critic = AttentionCritic(sa_sizes, hidden_dim=critic_hidden_dim, attend_heads=attend_heads)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=q_lr, weight_decay=1e-3)
        hard_update(self.highlevel_targetnet, self.highlevel_net)
        self.DuelDQN_optimizer = torch.optim.Adam(self.highlevel_net.parameters(), lr=high_lr, weight_decay=1e-3)
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.DQN_learn_step_counter = 0
        self.DQN_replace_target_iter = 100

    def choose_high_command(self, matrix_state, sparse_state, flag=False):
        matrix_state = torch.Tensor(matrix_state[np.newaxis, :])
        sparse_state = torch.Tensor(sparse_state[np.newaxis, :])
        high_action = self.highlevel_net.step(matrix_state, sparse_state)
        return high_action

    #  选择低层动作
    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def update_critic(self, sample_low):
        state, high_command, low_actions, rewards, next_state = sample_low
        next_low_actions = []
        next_log_pi = []
        for pi, command, ob in zip(self.target_policies, high_command, next_state):
            curr_next_ac, curr_log_pi = pi(ob, command, return_log_pi=True)
            next_low_actions.append(curr_next_ac)
            next_log_pi.append(curr_log_pi)
        next_critic_state = [torch.cat((t1, t2), dim=1) for t1, t2 in zip(next_state, high_command)]
        trgt_critic_in = list(zip(next_critic_state, next_low_actions))
        critic_state = [torch.cat((t1, t2), dim=1) for t1, t2 in zip(state, high_command)]
        critic_in = list(zip(critic_state, low_actions))
        with torch.no_grad():
            next_qs = self.target_critic(trgt_critic_in)
        critic_rets = self.critic(critic_in)
        q_loss = 0
        for a_i, nq, pq, log_pi in zip(range(self.nagents), next_qs, critic_rets, next_log_pi):
            target_q = (rewards[a_i].view(-1, 1) + self.gamma * nq)
            target_q -= 0.05 * log_pi
            q_loss += MSELoss(pq, target_q.detach())
        q_loss.backward()
        self.critic.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10 * self.nagents)
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

    def update_actors(self, sample_low):
        state, high_command, low_actions, rewards, next_state = sample_low
        samp_acs = []
        curr_log_pi = []
        disable_gradients(self.critic)
        for a_i, pi, ob, command in zip(range(self.nagents), self.policies, state, high_command):
            curr_ac, log_pi = pi(ob, command, return_log_pi=True)
            samp_acs.append(curr_ac)
            curr_log_pi.append(log_pi)
        critic_state = [torch.cat((t1, t2), dim=1) for t1, t2 in zip(state, high_command)]
        critic_in = list(zip(critic_state, samp_acs))
        critic_rets = self.critic(critic_in)
        for a_i, log_pi, (q) in zip(range(self.nagents), curr_log_pi, critic_rets):
            curr_agent = self.agents[a_i]
            pol_target = 0.05 * log_pi - q
            pol_loss = log_pi * (pol_target.detach()).mean()
            pol_loss.backward(torch.ones_like(pol_loss))
            enable_gradients(self.critic)
            torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
            curr_agent.policy_optimizer.step()
            curr_agent.policy_optimizer.zero_grad()

    def update_all_targets(self):
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)

    def prep_training(self):
        self.critic.train()
        self.target_critic.train()
        self.highlevel_net.train()
        self.highlevel_targetnet.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()

    def choose_low_action(self, matrix_state, high_command, curr, flag=False):
        low_actions = [a.step(state, command, curr=curr) for a, state, command in zip(self.agents, matrix_state, high_command)]
        return low_actions

    def prep_rollouts(self):
        for a in self.agents:
            a.policy.eval()

    def update_high_net(self, sample_high):
        if self.DQN_learn_step_counter % self.DQN_replace_target_iter == 0:
            self.highlevel_targetnet.load_state_dict(self.highlevel_net.state_dict())
        matrix_state, sparse_state, high_command, reward, next_state = sample_high  # buffer返回的是np array
        matrix_state = torch.Tensor(matrix_state)
        sparse_state = torch.Tensor(sparse_state)
        high_command = torch.Tensor(high_command).long().view(-1, num_agents*num_users)
        reward = torch.Tensor(reward)
        next_state = torch.Tensor(next_state)
        batch_size = matrix_state.shape[0]
        q_values = self.highlevel_net(matrix_state, sparse_state)
        q_values_re = q_values.reshape(batch_size, num_agents*num_users, 2)
        action_indices_expanded = high_command.unsqueeze(-1)
        q_value_action = q_values_re.gather(dim=2, index=action_indices_expanded)

        with torch.no_grad():
            next_q_values = self.highlevel_net(next_state, sparse_state)
            next_q_values = next_q_values.reshape(batch_size, num_agents*num_users, 2)
            max_next_q_values_indices = next_q_values.max(dim=2, keepdim=True)[1]
            next_q_target = self.highlevel_targetnet(next_state, sparse_state)
            next_q_target = next_q_target.reshape(batch_size, num_agents*num_users, 2)
            target_q_values_for_best_action = next_q_target.gather(dim=2, index=max_next_q_values_indices)
            target_q_values = reward.view(-1, 1, 1) + 0.95 * target_q_values_for_best_action
        loss = MSELoss(q_value_action, target_q_values)
        self.DuelDQN_optimizer.zero_grad()
        loss.backward()
        self.DuelDQN_optimizer.step()
        self.DQN_learn_step_counter += 1


















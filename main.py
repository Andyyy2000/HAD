import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from HAD.utils.utils import hard_update
from HAD.env.Env import CF_Mmimo
from HAD.algo.HAMARL import HAC
from HAD.buffer import LowBuffer, HighBuffer

num_agents = 15
num_users = 8
GAMMA = 0.95
TAU = 0.005
P_LR = 0.0001
Q_LR = 0.0001
high_LR = 0.001
re_scale = 10.
MEMORY_SIZE = 10000
num_episodes = 200
num_timeslot = 100
BATCH_SIZE = 128

def main():
    env = CF_Mmimo(num_agents=num_agents, num_users=num_users)
    model = HAC(sa_sizes=[(16, 8) for _ in range(num_agents)], gamma=GAMMA, tau=TAU, pi_lr=P_LR, q_lr=Q_LR, high_lr=high_LR, reward_scale=re_scale,
                state_dim1=(15, 8), state_dim2=8, action_dim=(15, 8),
                pol_hidden_dim=128, critic_hidden_dim=128, attend_heads=4, num_agents=num_agents,
                batch_size=BATCH_SIZE)
    low_buffer = LowBuffer(memory_size=MEMORY_SIZE, num_agents=num_agents, obs_dims=num_users, ac_dims=num_users)
    high_buffer = HighBuffer(memory_size=MEMORY_SIZE)

    t = 0
    low_rewards = []
    sum_rewards = []
    flag = False

    for ep_i in range(num_episodes):
        if t > MEMORY_SIZE:
            flag = True
        matrix_state, sparse_state = env.reset()
        matrix_state = np.abs(np.mean(matrix_state, axis=2))
        high_command = model.choose_high_command(matrix_state, sparse_state, flag)
        high_command_ = np.array(high_command.squeeze(dim=0))
        for et_i in range(num_timeslot):
            matrix_state_ = torch.Tensor(matrix_state[np.newaxis, :])
            actions = model.choose_low_action(matrix_state_, high_command, t, flag)
            actions = np.array(actions[0].detach())
            next_state, next_sparse, reward = env.step(high_command_, actions)  
            next_state = np.abs(np.mean(next_state, axis=2))
            low_buffer.restore_transition(matrix_state, high_command_, actions, reward, next_state)
            low_rewards.append(reward.sum())
            sum_rewards.append(reward.sum())
            if t % 10 == 0:
                high_reward = [sum(low_rewards) / len(low_rewards)]
                high_buffer.store_transition(matrix_state, sparse_state, high_command_, high_reward, next_state)
                next_high_command = model.choose_high_command(next_state, sparse_state)
                next_high_command_ = np.array(next_high_command.squeeze(dim=0))
                high_command = next_high_command
                high_command_ = next_high_command_
                low_rewards = []
            matrix_state = next_state
            t += 1
            if t > 200:
                model.prep_training()
                sample_low = low_buffer.sample(batch_size=BATCH_SIZE)
                model.update_critic(sample_low)
                model.update_actors(sample_low)
                model.update_all_targets()
            if t > 200 and t % 10:
                model.prep_training()
                sample_high = high_buffer.sample(batch_size=BATCH_SIZE)
                model.update_high_net(sample_high)
        sys_utility = np.mean(sum_rewards)
        sum_rewards = []


if __name__ == '__main__':
    main()



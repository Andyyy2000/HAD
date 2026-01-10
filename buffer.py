import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import random
from torch.autograd import Variable
from collections import deque


class HighBuffer(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)

    def store_transition(self, matrix_state, sparse_state, high_command, rewards, next_matrix):
        transition = (matrix_state, sparse_state, high_command, rewards, next_matrix)
        self.memory.append(transition)

    def sample(self, batch_size):
        curr_size = len(self.memory)
        actual_batch = min(batch_size, curr_size)
        sample = random.sample(self.memory, actual_batch)
        matrix, sparse_state, high_command, rewards, next_matrix = zip(*sample)
        matrix = np.array(matrix)
        sparse_state = np.array(sparse_state)
        high_command = np.array(high_command)
        rewards = np.array(rewards)
        next_matrix = np.array(next_matrix)
        return matrix, sparse_state, high_command, rewards, next_matrix

class LowBuffer(object):
    def __init__(self, memory_size, num_agents, obs_dims, ac_dims):
        self.memory_size = memory_size
        self.num_agents = num_agents
        self.obs_buffers = []
        self.command_buffers = []
        self.ac_buffers = []
        self.rew_buffers = []
        self.next_obs_buffers = []

        for _ in range(self.num_agents):
            self.obs_buffers.append(np.zeros((memory_size, obs_dims), dtype=np.float32))
            self.command_buffers.append(np.zeros((memory_size, ac_dims), dtype=np.float32))
            self.ac_buffers.append(np.zeros((memory_size, ac_dims), dtype=np.float32))
            self.rew_buffers.append(np.zeros((memory_size,), dtype=np.float32))
            self.next_obs_buffers.append(np.zeros((memory_size, obs_dims), dtype=np.float32))

        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0    # current index to write to (overwrite oldest data)

    def restore_transition(self, state, high_command, actions, rewards, next_state):
        nentries = 1
        if self.curr_i + nentries > self.memory_size:
            rollover = self.memory_size - self.curr_i
            for agent_i in range(self.num_agents):
                self.obs_buffers[agent_i] = np.roll(self.obs_buffers[agent_i], rollover, axis=0)
                self.command_buffers[agent_i] = np.roll(self.command_buffers[agent_i], rollover, axis=0)
                self.ac_buffers[agent_i] = np.roll(self.ac_buffers[agent_i], rollover, axis=0)
                self.rew_buffers[agent_i] = np.roll(self.rew_buffers[agent_i], rollover, axis=0)
                self.next_obs_buffers[agent_i] = np.roll(self.next_obs_buffers[agent_i], rollover, axis=0)
            self.curr_i = 0
            self.filled_i = self.memory_size

        for agent_i in range(self.num_agents):
            self.obs_buffers[agent_i][self.curr_i:self.curr_i + nentries] = state[agent_i, :]
            self.command_buffers[agent_i][self.curr_i:self.curr_i + nentries] = high_command[agent_i, :]
            self.ac_buffers[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i, :]
            self.rew_buffers[agent_i][self.curr_i:self.curr_i + nentries] = rewards[agent_i, :]
            self.next_obs_buffers[agent_i][self.curr_i:self.curr_i + nentries] = next_state[agent_i, :]
        self.curr_i += nentries
        if self.filled_i < self.memory_size:
            self.filled_i += nentries
        if self.curr_i == self.memory_size:
            self.curr_i = 0

    def sample(self, batch_size, norm_rews=True):
        inds = np.random.choice(np.arange(self.filled_i), size=batch_size, replace=False)
        cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffers[i][inds] - self.rew_buffers[i][:self.filled_i].mean()) / self.rew_buffers[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffers[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffers[i][inds]) for i in range(self.num_agents)],
                [cast(self.command_buffers[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffers[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffers[i][inds]) for i in range(self.num_agents)])






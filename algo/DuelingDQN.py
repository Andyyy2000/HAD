import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from HAD.utils.utils import hard_update

num_agents = 15

class DuelingDQN(nn.Module):
    def __init__(self, state_dim1, state_dim2, action_dim):
        super(DuelingDQN, self).__init__()
        self.dim = state_dim1
        self.fc1_matrix = nn.Linear(state_dim1[0] * state_dim1[1], 256)
        self.fc2_matrix = nn.Linear(state_dim2, 256)
        self.fc_merge = nn.Linear(512, 256)
        self.value_fc = nn.Linear(256, 1)
        self.advantage_fc = nn.Linear(256, action_dim[0] * action_dim[1] * 2)

    def forward(self, matrix_state, sparse_state):
        matrix_flat = matrix_state.view(matrix_state.size(0), -1)
        sparse_flat = sparse_state.view(sparse_state.size(0), -1)
        x_matrix = F.relu(self.fc1_matrix(matrix_flat))
        x_sparse = F.relu(self.fc2_matrix(sparse_flat))
        x = torch.cat([x_matrix, x_sparse], dim=1)
        x = F.relu(self.fc_merge(x))
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class High_DDQN(DuelingDQN):
    def __init__(self, state_dim1, state_dim2, action_dim, epsilon=0.1):
        super(High_DDQN, self).__init__(state_dim1, state_dim2, action_dim)
        self.state_dim1 = state_dim1
        self.state_dim2 = state_dim2
        self.action_dim = action_dim
        self.q_net = DuelingDQN(state_dim1, state_dim2, action_dim)
        self.epsilon = epsilon

    def forward(self, matrix_state, sparse_state):
        return self.q_net(matrix_state, sparse_state)

    def step(self, matrix_state, sparse_state):
        if random.random() < self.epsilon:
            action = np.random.randint(2, size=(15, 8))
            action = action[np.newaxis, :]
            for j in range(action.shape[2]):
                if np.sum(action[:, :, j]) == 0:
                    action[0, random.randint(0, 14), j] = 1
            return torch.from_numpy(action)
        else:
            with torch.no_grad():
                q_values = self.q_net(matrix_state, sparse_state)
                q_values = q_values.reshape(q_values.shape[0], 2, num_agents, -1)
                action = torch.argmax(q_values, dim=1)
                for j in range(action.shape[2]):
                    if torch.sum(action[:, :, j]) == 0:
                        action[0, random.randint(0, 14), j] = 1
                return action





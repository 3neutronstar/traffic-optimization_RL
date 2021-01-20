import torch
import os
import copy
from Agent.base import RLAlgorithm, merge_dict
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
DEFAULT_CONFIG = {
    'gamma': 0.99,
    'lr': 0.0001
}


class PolicyNet(nn.Module):
    def __init__(self, configs):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(configs['input_size'], 40)
        self.fc2 = nn.Linear(40, 30)
        self.fc3 = nn.Linear(30, configs['output_size'])
        self.running_loss = 0

    def forward(self, x):
        out = x
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class Trainer(RLAlgorithm):
    def __init__(self, configs):
        self.configs = merge_dict(configs, DEFAULT_CONFIG)
        self.model = PolicyNet(self.configs)
        self.gamma = self.configs['gamma']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.configs['lr'])
        self.saved_log_probs = []
        self.rewards = []
        self.eps = np.finfo(np.float32).eps.item()

    def get_action(self, state, reward):
        self.rewards.append(reward)
        state = state.float()
        print('state', state)
        probs = self.model(state)
        print('probs', probs)
        m = Categorical(probs)
        action = m.sample()
        print('action', action)
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def get_loss(self):
        return self.running_loss

    def update(self, done=False):
        R = 0  # Return
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:  # end기준으로 최근 reward부터 옛날로
            R = r + self.gamma * R
            returns.insert(0, R)  # 앞으로 내용 삽입(실제로는 맨뒤부터 삽입해서 t=0까지 삽입)
            # 내용순서는 t=0,1,2,3,4,...T)

        returns = torch.tensor(returns, device=self.configs['device'])
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()

        self.running_loss = policy_loss
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]

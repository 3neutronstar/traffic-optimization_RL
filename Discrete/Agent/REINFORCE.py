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
        self.fc1 = nn.Linear(configs['state_space'], 40)
        self.fc2 = nn.Linear(40, 30)
        self.fc3 = nn.Linear(30, configs['action_space'])
        self.running_loss = 0

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class Trainer(RLAlgorithm):
    def __init__(self, configs):
        self.configs = merge_dict(configs, DEFAULT_CONFIG)
        #self.configs=configs
        self.model = PolicyNet(self.configs)
        self.gamma = self.configs['gamma']
        self.lr=self.configs['lr']
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr)
        self.data=[]
        self.eps = torch.tensor(np.finfo(np.double).eps.item())
        self.lr_decay_rate=0.99

    def get_action(self, state):
        self.probability=self.model(state).float()
        m=Categorical(self.probability)
        action=m.sample()
        return action
    
    def put_data(self,item):
        self.data.append(item)

    def update(self, done=False):
        Return=0
        self.optimizer.zero_grad()
        for reward,prob in self.data[::-1]:
            Return=reward+self.gamma*Return #뒤에서 부터 부르면 reward에 최종 return값부터 불러짐
            loss= -torch.log(prob)*Return # for gradient ascend 원래 loss backward는 gradient descend이므로
            loss.backward()
        self.optimizer.step()
        self.data=[] # 지워버리기
    
    def get_prob(self):
        return self.probability
    
    def update_hyperparams(self, epoch):

        # decay learning rate
        if self.lr > 0.01*self.lr:
            self.lr = self.lr_decay_rate*self.lr
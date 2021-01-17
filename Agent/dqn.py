from Agent.base import RLAlgorithm
import torch
from torch import nn
import torch.nn.functional as f
import numpy as np
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    def __init__(self, configs):
        super(QNetwork, self).__init__()
        self.configs = configs
        self.input_size = self.configs['input_size']
        self.output_size = self.configs['output_size']
        self.configs['fc_net'] = [40, 30]

        # build nn
        self.fc = list()
        for i, layers in enumerate(self.configs['fc_net']):
            if i == 1:
                self.fc.append(nn.Linear(self.input_size, layers))
            elif i == len(self.configs['fc_net']):
                self.fc.append(nn.Linear(before_layers, layers))
                self.fc.append(nn.Linear(layers, self.output_size))
            else:
                self.fc.append(nn.Linear(before_layers, layers))
            before_layers = layers

    def forward(self, state):
        x = state
        for _, fc in enumerate(self.fc):
            x = f.relu(fc(x))
        return x  # q value


class Trainer(RLAlgorithm):
    def __init__(self, optimizer, configs):
        super().__init__(configs)
        configs['batch_size'] = 32
        configs['experience_replay_size'] = 1e5
        self.configs = configs
        self.input_size = self.configs['input_size']
        self.output_size = self.configs['output_size']
        self.action_space = self.configs['action_space']
        self.mainQNetwork = QNetwork(self.configs)
        self.targetQNetwork = QNetwork(self.configs)
        self.configs['experience_replay_size'] = 20000
        for param in self.targetQNetwork.parameters():  # target Q는 hold
            param.requires_grad = False
        self.epsilon = 0.5
        self.mainQNetwork.train()  # train모드로 설정
        learning_rate = 0.0001
        self.running_loss = 0
        self.optimizer = optimizer
        self.experience_replay = deque(self.configs['experience_replay_size'])
        self.batch_size = self.configs['batch_size']

        if self.configs['mode'] == 'train':
            self.mainQNetwork.train()
        elif self.configs['mode'] == 'test':
            self.mainQNetwork.eval()

    def get_action(self, state):
        self.optimizer.zero_grad()
        self.Q = self.mainQNetwork(state)
        self.targetQ = self.targetQNetwork(state)
        self.targetQ.freeze
        self.criterion = nn.MSELoss()

        _, action = torch.max(self.Q, dim=1)
        action = action.data[0].item()
        # fc net 거쳐왔으므로 dimension이 1임 그러면 max를 거치면 하나의 요소만 나옴 but list라서 0으로 뽑아내는것

        if np.random.rand(1) < self.epsilon:
            return action
        else:
            return action

    def get_loss(self):
        return self.running_loss

    def target_update(self):
        self.targetQNetwork.load_state_dict(self.mainQNetwork.state_dict())

    def save_replay(self, state, action, reward, next_state):
        self.experience_replay.append((state, action, reward, next_state))

    def update(self, done=False):
        batch_sampled_data = random.sample(
            list(self.experience_replay), self.batch_size)
        for i, state, action, reward, next_state in enumerate(batch_sampled_data):
            if done:
                targetQvalue = reward
            else:
                targetQvalue = reward+self.gamma * \
                    torch.max(self.targetQNetwork(next_state)[0])  # 맞는지 확인

            targetQValue = self.targetQNetwork(
                state)  # MSE에서 Mean 값에 맞게는 어떻게 할지?
            QValue = self.mainQNetwork(state)

        loss = self.criterion(QValue, targetQValue)
        loss.backward()
        self.optimizer.step()
        self.running_loss += loss.item()

        if self.epsilon > 0.2:
            self.epsilon *= 0.97  # decay rate

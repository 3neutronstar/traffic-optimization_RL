from Agent.base import RLAlgorithm, ReplayMemory, merge_dict
import torch
from torch import nn
import torch.nn.functional as f
import numpy as np
import torch.optim as optim
import random
import os
from collections import namedtuple
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

DEFAULT_CONFIG = {
    'gamma': 0.99,
    'tau': 0.995,
    'batch_size': 32,
    'experience_replay_size': 1e5,
    'epsilon': 0.5,
    'epsilon_decay_rate': 0.95
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, configs):
        super(QNetwork, self).__init__()
        self.configs = merge_dict(configs, DEFAULT_CONFIG)
        self.input_size = input_size
        self.output_size = output_size
        self.configs['fc_net'] = [40, 30]

        # build nn
        self.fc1 = nn.Linear(self.input_size, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16, self.output_size)
        # self.fc4 = nn.Linear(30, self.output_size)

    def forward(self, x):
        x = x.float()
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = self.fc4(x)
        #x = self.fc3(x)
        #x = f.softmax(self.fc4(x), dim=0)
        return x  # q value


class Trainer(RLAlgorithm):
    def __init__(self, configs):
        super().__init__(configs)
        self.configs = merge_dict(configs, DEFAULT_CONFIG)
        self.input_size = self.configs['input_size']
        self.output_size = self.configs['output_size']
        self.action_space = self.configs['action_space']
        self.gamma = self.configs['gamma']
        self.epsilon = self.configs['epsilon']
        self.criterion = nn.MSELoss()
        self.configs['lr'] = 0.001
        self.lr = self.configs['lr']
        self.epsilon_decay_rate = self.configs['epsilon_decay_rate']
        self.experience_replay = ReplayMemory(
            self.configs['experience_replay_size'])
        self.batch_size = self.configs['batch_size']

        if self.configs['model'] == 'FRAP':
            from Agent.Model.FRAP import FRAP
            model = FRAP(self.input_size, self.output_size)
            model.add_module(
                QNetwork(self.input_size, self.output_size, self.configs))
        else:
            model = QNetwork(self.input_size, self.output_size,
                             configs)  # 1개 네트워크용
        model.to(self.configs['device'])
        print(model)
        self.mainQNetwork = deepcopy(model).to(self.configs['device'])
        self.targetQNetwork = deepcopy(model).to(self.configs['device'])
        self.targetQNetwork.load_state_dict(self.mainQNetwork.state_dict())
        self.optimizer = optim.Adam(
            self.mainQNetwork.parameters(), lr=self.lr)
        self.targetQNetwork.eval()
        self.mainQNetwork.train()  # train모드로 설정
        self.rewards = []
        self.action = tuple()
        self.running_loss = 0
        if self.configs['mode'] == 'train':
            self.mainQNetwork.train()
        elif self.configs['mode'] == 'test':
            self.mainQNetwork.eval()

    def get_action(self, state, reward):
        self.rewards.append(reward)
        sample = random.random()
        # self.Q=self.Q.reshape(2,8) # Q value를 max값 선택하게
        if sample > self.epsilon:
            with torch.no_grad():
                action = torch.max(self.mainQNetwork(state), dim=1)[
                    1].view(1, 1)  # 가로로
                self.action+=tuple(action)
            return action
        else:
            action = torch.tensor([random.randint(0, 1)
                                   for i in range(self.action_space)], device=self.configs['device']).view(1, 1)
            self.action+=tuple(action)
            return action

    def target_update(self):
        # state_dict=self.targetQNetwork.state_dict()*self.configs['tau']+(1-self.configs['tau'])*self.mainQNetwork.state_dict()
        # self.targetQNetwork.load_state_dict(state_dict)
        # Hard Update
        self.targetQNetwork.load_state_dict(self.mainQNetwork.state_dict())

    def save_replay(self, state, action, reward, next_state):
        self.experience_replay.push(
            state, action, reward, next_state)

    def update(self, done=False):
        if len(self.experience_replay) < self.configs['batch_size']:
            return

        transitions = self.experience_replay.sample(self.configs['batch_size'])
        batch = Transition(*zip(*transitions))

        # 최종 상태가 아닌 마스크를 계산하고 배치 요소를 연결합니다.
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.configs['device'], dtype=torch.bool)

        #non_final_mask.reshape(self.configs['batch_size'], 1)
        # non_final_next_states = torch.tensor([s for s in batch.next_state
        #                                       if s is not None]).reshape(-1, 1)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None], dim=0)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action, dim=0)  # 안쓰지만 배치함

        # reward_batch = torch.cat(torch.tensor(batch.reward, dim=0)
        reward_batch = torch.tensor(batch.reward).to(self.configs['device'])
        # .reshape(
        #     self.configs['batch_size']).to(self.configs['device'])

        # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 칼럼을 선택한다.

        state_action_values = self.mainQNetwork(
            state_batch).gather(1, action_batch)  # for 3D
        # state_action_values = self.mainQNetwork(
        #     state_batch)
        # .max(1)[0].clone().float().unsqueeze(1)

        # 모든 다음 상태를 위한 V(s_{t+1}) 계산
        next_state_values = torch.zeros(
            self.configs['batch_size'], device=self.configs['device'], dtype=torch.float)

        next_state_values[non_final_mask] = self.targetQNetwork(
            non_final_next_states).max(1)[0].detach().to(self.configs['device'])  # .to(self.configs['device'])  # 자신의 Q value 중에서max인 value를 불러옴

        # 기대 Q 값 계산
        expected_state_action_values = (
            next_state_values * self.configs['gamma']) + reward_batch

        # loss 계산
        loss = self.criterion(state_action_values,
                              expected_state_action_values.unsqueeze(1))
        self.running_loss += loss
        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.mainQNetwork.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_hyperparams(self, epoch):
        # decay rate (epsilon greedy)
        if self.epsilon > 0.005:
            self.epsilon *= self.epsilon_decay_rate

        # decay learning rate
        if self.lr > 0.01*self.lr:
            self.lr = 0.99*self.lr

    def save_weights(self, name):
        torch.save(self.mainQNetwork.state_dict(), os.path.join(
            self.configs['current_path'], 'training_data', 'model', name+'.h5'))
        torch.save(self.targetQNetwork.state_dict(), os.path.join(
            self.configs['current_path'], 'training_data', 'model', name+'_target.h5'))

    def load_weights(self, name):
        self.mainQNetwork.load_state_dict(torch.load(os.path.join(
            self.configs['current_path'], 'training_data', 'model', name+'.h5')))
        self.mainQNetwork.eval()

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('episode/loss', self.running_loss/self.configs['max_steps'],
                          self.configs['max_steps']*epoch)  # 1 epoch마다
        writer.add_scalar('hyperparameter/lr', self.lr,
                          self.configs['max_steps']*epoch)
        writer.add_scalar('hyperparameter/epsilon',
                          self.epsilon, self.configs['max_steps']*epoch)

        action_distribution = torch.cat(self.action, 0)
        writer.add_histogram('hist/episode/action_distribution', action_distribution,
                             epoch)  # 1 epoch마다
        self.action = tuple()
        # clear
        self.running_loss = 0

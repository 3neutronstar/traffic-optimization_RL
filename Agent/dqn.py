from Agent.base import RLAlgorithm
import torch
from torch import nn
import torch.nn.functional as f
import numpy as np
import torch.optim as optim
import random

from collections import deque,namedtuple
from copy import deepcopy

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """전환 저장"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[int(self.position)] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    def __init__(self, input_size,output_size,configs):
        super(QNetwork, self).__init__()
        self.configs = configs
        self.input_size = input_size
        self.output_size = output_size
        self.configs['fc_net'] = [40, 30]

        # build nn
        self.fc = list()
        self.fc1=nn.Linear(self.input_size,40)
        self.fc2=nn.Linear(40,30)
        self.fc3=nn.Linear(30,self.output_size)

    def forward(self, state):
        x = state
        for _, fc in enumerate(self.fc):
            x = f.relu(fc(x))
        return x  # q value


class Trainer(RLAlgorithm):
    def __init__(self, configs):
        super().__init__(configs)
        configs['batch_size'] = 32
        configs['experience_replay_size'] = 1e5
        configs['input_size']=16
        configs['output_size']=8
        configs['action_space']=8
        self.configs = configs
        self.input_size = self.configs['input_size']
        self.output_size = self.configs['output_size']
        self.action_space = self.configs['action_space']
        self.gamma=self.configs['gamma']
        self.epsilon = 0.5
        self.criterion = nn.MSELoss()
        self.experience_replay = ReplayMemory(self.configs['experience_replay_size'])
        self.batch_size = self.configs['batch_size']

        if self.configs['model']=='FRAP':
            from Agent.Model.FRAP import FRAP
            model=FRAP(self.input_size,self.output_size)
            model.add_module(QNetwork(self.input_size,self.output_size,self.configs))
            print(model)
        else:
            model=QNetwork(16,8,configs)# 1개 네트워크용
        model.to(configs['device'])
        print(model)
        self.mainQNetwork = deepcopy(model)
        self.targetQNetwork = deepcopy(model)
        self.optimizer = optim.Adam(
            self.mainQNetwork.parameters(), lr=configs['learning_rate'])
        self.configs['experience_replay_size'] = 20000
        for param in self.targetQNetwork.parameters():  # target Q는 hold
            param.requires_grad = False
        self.mainQNetwork.train()  # train모드로 설정

        self.running_loss = 0
        if self.configs['mode'] == 'train':
            self.mainQNetwork.train()
        elif self.configs['mode'] == 'test':
            self.mainQNetwork.eval()

    def get_action(self, state):
        sample=random.random()
        with torch.no_grad():
            self.Q=self.mainQNetwork(state)
        _,action=torch.max(self.Q,dim=1)
        print(action)
        if sample < self.epsilon:
                return action
        else:
            action=torch.tensor([random.randint(0,1) for i in range(8)],device=self.configs['device'])
            return action

    def get_loss(self):
        return self.running_loss

    def target_update(self):
        self.targetQNetwork.load_state_dict(self.mainQNetwork.state_dict())

    def save_replay(self, state, action, reward, next_state):
        self.experience_replay.push(state,action,reward,next_state)

    def update(self, done=False):
        if len(self.experience_replay)<self.configs['batch_size']:
            return
        
        transitions = self.experience_replay.sample(self.configs['batch_size'])
        # Transpose the batch (자세한 설명을 위해 http://stackoverflow.com/a/19343/3343043 를 보십시오).
        batch = Transition(*zip(*transitions))

        # 최종 상태가 아닌 마스크를 계산하고 배치 요소를 연결합니다.
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.configs['device'], dtype=torch.uint8)
        non_final_next_states = batch.next_state# torch.cat([s for s in torch.tensor(batch.next_state)
                                #                    if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        

        # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 칼럼을 선택한다.

        state_action_values = self.mainQNetwork(state_batch).gather(1,action_batch)

        # 모든 다음 상태를 위한 V(s_{t+1}) 계산
        next_state_values = torch.zeros(self.configs['batch_size'], device=self.configs['device'])
        next_state_values[non_final_mask] = self.targetQNetwork(non_final_next_states).max(1)[0].detach()
        # 기대 Q 값 계산
        expected_state_action_values = (next_state_values * self.configs['gamma']) + reward_batch

        # loss 계산
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.mainQNetwork.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.epsilon > 0.2:
            self.epsilon *= 0.97  # decay rate

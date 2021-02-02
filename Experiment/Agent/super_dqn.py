import torch
from torch import nn
import torch.nn.functional as f
import numpy as np
import torch.optim as optim
import random
import os
from collections import namedtuple
from copy import deepcopy
from Agent.base import RLAlgorithm, ReplayMemory, merge_dict, hard_update
from torch.utils.tensorboard import SummaryWriter

DEFAULT_CONFIG = {
    'gamma': 0.99,
    'tau': 0.995,
    'batch_size': 32,
    'experience_replay_size': 1e5,
    'epsilon': 0.9,
    'epsilon_decay_rate': 0.98,
    'fc_net': [32, 32, 16],
    'lr': 5e-5,
    'lr_decay_rate': 0.98,
    'target_update_period': 5,
    'final_epsilon': 0.0005,
    'final_lr': 1e-7,
}

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class QNetwork(nn.Module):
    def __init__(self, input_size, rate_output_size, time_output_size, configs):
        super(QNetwork, self).__init__()
        self.configs = configs
        self.input_size = input_size
        self.rate_output_size = rate_output_size
        self.num_agent = len(configs['tl_rl_list'])

        # build nn 증감
        self.fc1 = nn.Linear(self.input_size, self.configs['fc'][0])
        self.fc2 = nn.Linear(self.configs['fc'][0], self.configs['fc'][1])
        self.fc3 = nn.Linear(self.configs['fc'][1], self.configs['fc'][2])
        self.fc4 = nn.Linear(self.configs['fc'][2], self.rate_output_size)
        # 증감의 크기
        # +1은 증감에서의 argmax value
        self.fc_y1 = nn.Linear(self.input_size+1, self.configs['fc'][0])
        self.fc_y2 = nn.Linear(self.configs['fc'][0], self.configs['fc'][1])
        self.fc_y3 = nn.Linear(self.configs['fc'][1], self.configs['fc'][2])
        self.fc_y4 = nn.Linear(self.configs['fc'][2], self.time_output_size)

    def forward(self, input_x):
        # 증감
        x = f.leaky_relu(self.fc1(input_x))
        x = f.leaky_relu(self.fc2(x))
        x = f.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        # 증감의 크기
        # argmax(x)값을 구해서 넣기
        y = torch.cat(input_x, x.max(
            1)[1].detach().view(-1, self.num_agent, 1), dim=1)
        y = f.leaky_relu(self.fc_y1(y))
        y = f.leaky_relu(self.fc_y2(y))
        y = f.leaky_relu(self.fc_y3(y))
        y = self.fc_y4(y)
        return x, y  # q value


class SuperQNetwork(nn.Module):
    def __init__(self, input_size, output_size, configs):
        super(SuperQNetwork, self).__init__()
        self.configs = configs
        self.input_size = input_size
        self.output_size = output_size
        self.num_agent = len(self.configs['tl_rl_list'])
        self.fc1 = nn.Linear(
            self.input_size, self.configs['state_space']*2*self.num_agent)
        self.fc2 = nn.Linear(
            self.configs['state_space']*2*self.num_agent, self.configs['state_space']*2*self.num_agent)
        self.fc3 = nn.Linear(
            self.configs['state_space']*2*self.num_agent, self.configs['state_space']*1*self.num_agent)
        self.fc4 = nn.Linear(
            self.configs['state_space']*1*self.num_agent, self.output_size)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = self.fc4(x).view(-1, self.num_agent, self.output_size)
        return x


class Trainer(RLAlgorithm):
    def __init__(self, configs):
        super().__init__(configs)
        os.mkdir(os.path.join(
            self.configs['current_path'], 'training_data', self.configs['time_data'], 'model'))
        self.configs = merge_dict(configs, DEFAULT_CONFIG)
        self.num_agent = len(self.configs['tl_rl_list'])
        self.state_space = self.configs['state_space']
        # rate action space
        self.rate_action_space = self.configs['rate_action_space']
        self.time_action_space = configs['action_spacetime_action_space']
        self.action_size = self.configs['action_size']
        self.gamma = self.configs['gamma']
        self.epsilon = self.configs['epsilon']
        self.criterion = nn.MSELoss()
        self.lr = self.configs['lr']
        self.lr_decay_rate = self.configs['lr_decay_rate']
        self.epsilon_decay_rate = self.configs['epsilon_decay_rate']
        self.experience_replay = ReplayMemory(
            self.configs['experience_replay_size'])
        self.batch_size = self.configs['batch_size']
        self.running_loss = 0

        # NN composition
        self.mainSuperQNetwork = SuperQNetwork(
            self.configs['state_space']*self.num_agent, self.num_agent*self.configs['state_space']/2, self.configs)
        self.targetSuperQNetwork = SuperQNetwork(
            self.configs['state_space']*self.num_agent, self.num_agent*self.configs['state_space']/2, self.configs)
        self.mainQNetwork[self.num_agent] = [QNetwork(
            self.num_agent*self.configs['state_space']/2, self.rate_action_space, self.time_action_space, self.configs)*self.num_agent]
        self.targetQNetwork[self.num_agent] = [QNetwork(
            self.num_agent*self.configs['state_space']/2, self.rate_action_space, self.time_action_space, self.configs)*self.num_agent]

        # hard update
        hard_update(self.targetSuperQNetwork, self.mainSuperQNetwork)
        hard_update(self.targetQNetwork, self.mainQNetwork)
        for _ in range(self.num_agent):
            self.optimizer = optim.Adam(
                self.mainSuperQNetwork.parameters()+self.mainQNetwork.parameters(), lr=self.lr)

        self.action = tuple()
        # Network
        print("========SUPER NETWORK==========\n", self.mainSuperQNetwork)
        print("========NETWORK==========\n")
        for i in range(self.num_agent):
            print(self.mainQNetwork[i])

    def get_action(self, state, mask):
        # with torch.no_grad():
        #     action=torch.max(self.mainQNetwork(state),dim=2)[1].view(-1,self.num_agent,self.action_size)
        #     for i in range(self.num_agent):
        #         if random.random()<self.epsilon:
        #             action[0][i]=random.randint(0,7)

        # 전체를 날리는 epsilon greedy
        if random.random() > self.epsilon:  # epsilon greedy
            with torch.no_grad():
                actions = tuple()
                obs = self.mainSuperQNetwork(state)
                for i, mainQNetwork in enumerate(self.mainQNetwork):
                    action = torch.max(mainQNetwork(obs), dim=2)[
                        1].view(-1, self.num_agent, self.action_size)
                    actions += tuple(action)  # dim 2에서 고름
                actions = torch.cat(actions, dim=1)
                # agent가 늘어나면 view(agents,action_size)
                self.action += tuple(actions)  # 기록용
            return actions
        else:
            rate_action = torch.tensor([random.randint(0, self.rate_action_space-1)  # 여기서 3일 때, phase 4 7일때 phase8
                                        for i in range(self.num_agent)], device=self.configs['device']).view(-1, self.num_agent, self.rate_action_space)
            time_action = torch.tensor(
                [random.randint(0, self.configs['time_action_space']-1) for i in range(self.num_agent)]).view(1, -1)
            actions = torch.cat((rate_action, time_action), dim=1)
            self.action += tuple(actions)  # 기록용
            return actions

    def target_update(self):
        # Hard Update
        self.targetQNetwork.load_state_dict(self.mainQNetwork.state_dict())
        self.targetSuperQNetwork.load_state_dict(
            self.mainSuperQNetwork.state_dict())

    def save_replay(self, state, action, reward, next_state):
        self.experience_replay.push(
            state, action, reward, next_state)  # asynchronous하게 저장하고 불러오기

    def update(self, done=False):  # 각 agent마다 시행하기 # agent network로 돌아가서 시행 그러면될듯?
        if len(self.experience_replay) < self.configs['batch_size']:
            return

        transitions = self.experience_replay.sample(self.configs['batch_size'])
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state))*self.num_agent, device=self.configs['device'], dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None], dim=1).view(-1, self.num_agent*self.state_space)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)  # batch x num_agent

        state_action_values = self.mainQNetwork(
            state_batch).gather(2, action_batch).view(-1, self.num_agent, self.action_size)  # batchxagent

        # 1차원으로 눌러서 mapping 하고
        next_state_values = torch.zeros(
            (self.configs['batch_size']*self.num_agent), device=self.configs['device'], dtype=torch.float)  # batch*agent
        next_state_values[non_final_mask] = self.targetQNetwork(
            non_final_next_states).max(dim=2)[0].detach().to(self.configs['device']).view(-1)  # .to(self.configs['device'])  # 자신의 Q value 중에서max인 value를 불러옴
        # 다시 원래 차원으로 돌리기
        next_state_values = next_state_values.view(
            -1, self.num_agent, self.action_size)
        # 기대 Q 값 계산
        expected_state_action_values = (
            next_state_values * self.configs['gamma']) + reward_batch
        # loss 계산
        loss = self.criterion(state_action_values.unsqueeze(2),
                              expected_state_action_values.unsqueeze(2))  # 행동이 하나이므로 2차원에 대해 unsqueeze
        self.running_loss += loss/self.configs['batch_size']
        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.mainQNetwork.parameters():
            param.grad.data.clamp_(-1, 1)  # 값을 -1과 1로 한정시켜줌 (clipping)
        self.optimizer.step()

    def update_hyperparams(self, epoch):
        # decay rate (epsilon greedy)
        if self.epsilon > self.configs['final_epsilon']:
            self.epsilon *= self.epsilon_decay_rate

        # decay learning rate
        if self.lr > self.configs['final_lr']:
            self.lr = self.lr_decay_rate*self.lr

    def save_weights(self, name):
        torch.save(self.mainQNetwork.state_dict(), os.path.join(
            self.configs['current_path'], 'training_data', self.configs['time_data'], 'model', name+'.h5'))
        torch.save(self.targetQNetwork.state_dict(), os.path.join(
            self.configs['current_path'], 'training_data', self.configs['time_data'], 'model', name+'_target.h5'))

    def load_weights(self, name):
        self.mainQNetwork.load_state_dict(torch.load(os.path.join(
            self.configs['current_path'], 'training_data', self.configs['time_data'], 'model', name+'.h5')))
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

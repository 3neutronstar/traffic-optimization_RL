import torch
import gym
import os
import random
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


DEFAULT_CONFIG = {
    'gamma': 0.99,
    'lr': 0.01,
    'decay_rate': 0.98,
    'actior_lr': 0.001,
    'critic_lr': 0.001,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
}


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, state_space, configs):
        self.states = torch.zeros(num_steps+1, num_processes, state_space)
        self.masks = torch.ones(num_steps+1, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()
        self.gamma = configs['gamma']
        self.returns = torch.zeros(num_steps+1, num_processes, 1)
        self.index = 0

    def insert(self, state, action, reward, mask):
        self.states[self.index+1].copy_(state)
        self.masks[self.index+1].copy_(mask)
        self.actions[self.index].copy_(action)
        self.rewards[self.index].copy_(reward)

        self.index = (self.index+1) % NUM_ADVANCED_STEP  # index값 업데이트

    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.gamma*self.returns[ad_step+1] * \
                self.masks[ad_step+1]+self.rewards[ad_step]


class Net(nn.Module):
    def __init__(self, configs):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(configs['state_space'], 32)
        self.fc2 = nn.Linear(32, 32)
        self.actor = nn.Linear(32, configs['action_space'])
        self.critic = nn.Linear(32, 1)  # Qvalue
        self.running_loss = 0

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        critic_output = self.critic(x)
        actor_output = self.actor(x)
        return critic_output, actor_output

    def act(self, x):
        _, actor_output = self(x)  # forward
        # dim=1 이면 같은 행(행동 종류에 대해서 sofmax)
        action_prob = F.softmax(actor_output, dim=1)
        action = action_prob.multinomial(num_samples=1)
        #action = Categorical(action_prob).sample()
        return action

    def get_state_value(self, x):
        state_value, _ = self(x)
        return state_value

    def evaluate_action(self, x, actions):  # x가 process수 만큼 들어옴
        state_value, actor_output = self(x)
        log_prob = F.log_softmax(actor_output, dim=1)
        # batch중에서 자신이 선택했던 action의 위치의 값을 반환Q(s,a')
        action_log_prob = log_prob.gather(1, actions)

        probs = F.softmax(actor_output, dim=1)
        entropy = -(log_prob*probs).sum(-1).mean()  # 전체 process수 의 평균
        return state_value, action_log_prob, entropy


class Trainer(RLAlgorithm):
    def __init__(self, configs):
        self.configs = merge_dict(configs, DEFAULT_CONFIG)
        self.actor_critic = Net(configs)
        self.actor_critic.train()
        self.gamma = self.configs['gamma']
        self.lr = self.configs['lr']
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=self.lr)
        self.action_size = self.configs['action_size']
        self.running_loss = 0
        self.state_space = self.configs['state_space']
        self.rollouts = RolloutStorage(
            NUM_ADVANCED_STEP, NUM_PROCESSES, self.state_space, self.configs)

    def get_action(self, state):
        action = self.actor_critic.act(state)
        return action

    def update(self):
        state_space = self.rollouts.states.size()[2:]
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        state_values, action_log_prob, entropy = self.actor_critic.evaluate_action(
            self.rollouts.states[:-1].view(-1, self.state_space), self.rollouts.actions.view(-1, self.action_size))

        state_values = state_values.view(num_steps, num_processes, 1)
        action_log_prob = action_log_prob.view(num_steps, num_processes, 1)

        advantages = self.rollouts.returns[:-1] - state_values

        value_loss = advantages.pow(2).mean()  # MSE
        action_gain = (action_log_prob*advantages.detach()
                       ).mean()  # detach로 상수화
        total_loss = (value_loss*value_loss_coef - action_gain -
                      entropy*entropy_coeff)  # cross entropy

        self.optimizer.zero_grad()

        total_loss.backward()
        # 경사 clipping, 너무 한번에 크게 변화하지 않도록
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
        self.optimizer.step()

    def update_hyperparams(self, epoch):

        # decay learning rate
        if self.lr > 0.01*self.configs['lr']:
            self.lr = self.lr_decay_rate*self.lr

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('episode/loss', self.running_loss/self.configs['max_steps'],
                          self.configs['max_steps']*epoch)  # 1 epoch마다
        writer.add_scalar('hyperparameter/lr', self.lr,
                          self.configs['max_steps']*epoch)

        action_distribution = torch.cat(self.action, 0)
        writer.add_histogram('hist/episode/action_distribution', action_distribution,
                             epoch)  # 1 epoch마다
        self.action = tuple()
        # clear
        self.running_loss = 0

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from Agent.base import RLAlgorithm, merge_dict, ReplayMemory


DEFAULT_CONFIG = {
    'gamma': 0.99,
    'lr': 0.001,
    'actor_layers': [30, 30],
    'critic_layers': [30, 30],
    'num_sgd_iter': 4,
    'eps_clip': 0.2,
    'lr_decay_rate': 0.99,
    'update_period': 5,
    'vf_loss_coeff':1.0,
    'entropy_coeff':0.01,

}


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]


class Net(nn.Module):
    def __init__(self, memory, configs):
        super(Net, self).__init__()
        self.memory = memory
        self.actor = nn.Sequential(
            nn.Linear(configs['state_space'], configs['actor_layers'][0]),
            nn.Tanh(),
            nn.Linear(configs['actor_layers'][0], configs['actor_layers'][0]),
            nn.Tanh(),
            nn.Linear(configs['actor_layers'][1], configs['action_space']),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(configs['state_space'], configs['critic_layers'][0]),
            nn.Tanh(),
            nn.Linear(configs['critic_layers'][0],
                      configs['critic_layers'][1]),
            nn.Tanh(),
            # Q value 는 1개, so softmax 사용 x
            nn.Linear(configs['critic_layers'][1], 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        distributions = Categorical(action_probs)
        action = distributions.sample()

        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(distributions.log_prob(action))

        return action

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        distributions = Categorical(action_probs)

        action_logprobs = distributions.log_prob(action)
        distributions_entropy = distributions.entropy()

        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), distributions_entropy


class Trainer(RLAlgorithm):
    def __init__(self, configs):
        self.memory = Memory()
        self.configs = merge_dict(configs, DEFAULT_CONFIG)
        self.gamma = self.configs['gamma']
        self.eps_clip = self.configs['eps_clip']
        self.lr = self.configs['lr']
        self.lr_decay_rate = self.configs['lr_decay_rate']
        self.num_sgd_iter = self.configs['num_sgd_iter']
        self.vf_loss_coeff=self.configs['vf_loss_coeff']
        self.model = Net(self.memory, self.configs).to(self.configs['device'])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr)
        self.entropy_coeff=self.configs['entropy_coeff']
        self.model_old = Net(self.memory, self.configs).to(
            self.configs['device'])
        self.model_old.load_state_dict(self.model.state_dict())
        self.running_loss = 0
        self.action = tuple()
        self.criterion = nn.MSELoss()

    def get_action(self, state):
        action = self.model_old.act(state)
        self.action += tuple(action.view(1, 1))
        return action

    def update(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(
            self.configs['device'])
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(self.memory.states).to(
            self.configs['device']).detach()
        old_actions = torch.stack(self.memory.actions).to(
            self.configs['device']).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(
            self.configs['device']).detach()

        # Optimize model for K epochs:
        for _ in range(self.num_sgd_iter):  # k번 업데이트
            # Evaluating old actions and values :
            logprobs, state_values, distributions_entropy = self.model.evaluate(
                old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            # old쪽은 중복(old action쪽과)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()  # detach이유는 중복되기 때문
            surr1 = ratios * advantages  # ratio가 Conservative policy iteration
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + self.vf_loss_coeff * \
                self.criterion(state_values, rewards) - \
                self.entropy_coeff*distributions_entropy  # criterion 부분은 value loss이며 reward는 cumulative이므로 사용가능
            # 마지막항은  entropy bonus항

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.running_loss += loss.mean()
        # Copy new weights into old model:
        self.model_old.load_state_dict(self.model.state_dict())
        self.memory.clear_memory()

    def update_hyperparams(self, epoch):
        # decay learning rate
        if self.lr > 0.05*self.configs['lr']:
            self.lr = self.lr_decay_rate*self.lr

    def save_weights(self, name):
        torch.save(self.model.state_dict(), os.path.join(
            self.configs['current_path'], 'training_data', 'model', name+'.h5'))
        torch.save(self.model_old.state_dict(), os.path.join(
            self.configs['current_path'], 'training_data', 'model', name+'_old.h5'))

    def load_weights(self, name):
        self.model_old.load_state_dict(torch.load(os.path.join(
            self.configs['current_path'], 'training_data', 'model', name+'.h5')))
        self.model_old.eval()

    def update_tensorboard(self, writer, epoch):
        if epoch % self.configs['update_period'] == 0:  # 5마다 업데이트
            writer.add_scalar('episode/total_loss', self.running_loss/self.configs['max_steps'],
                                self.configs['max_steps']*epoch)  # 1 epoch마다
            self.running_loss = 0
        writer.add_scalar('hyperparameter/lr', self.lr,
                          self.configs['max_steps']*epoch)
        action_distribution = torch.cat(self.action, 0)
        writer.add_histogram('hist/episode/action_distribution', action_distribution,
                             epoch)  # 1 epoch마다
        self.action = tuple()
        # clear

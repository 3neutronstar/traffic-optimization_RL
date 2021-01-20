import torch
from torch import nn
import copy
import random
from collections import namedtuple
from copy import deepcopy
import os


class RLAlgorithm():
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

    def get_action(self, state):
        '''
        return action (torch Tensor (1,action_space))
        상속을 위한 함수
        '''
        raise NotImplementedError

    def get_loss(self):
        '''
        return loss.item()
        반드시 get_action뒤에 사용
        상속을 위한 함수
        '''
        raise NotImplementedError


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


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


def merge_dict(d1, d2):
    merged = copy.deepcopy(d1)
    for key in d2.keys():
        if key in merged.keys():
            raise KeyError
        merged[key] = d2[key]
    return merged

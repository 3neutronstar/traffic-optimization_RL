import torch
from torch import nn


class RLAlgorithm():
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

    def get_action(self):
        '''
        상속을 위한 함수
        '''
        raise NotImplementedError

    def get_loss(self):
        '''
        반드시 get_action뒤에 사용
        상속을 위한 함수
        '''
        raise NotImplementedError

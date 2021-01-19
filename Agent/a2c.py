import torch
import os
import copy
from base import RLAlgorithm, merge_dict
DEFAULT_CONFIG = {

}


class Trainer(RLAlgorithm):
    def __init__(self, configs):
        self.configs = merge_dict(configs, DEFAULT_CONFIG)

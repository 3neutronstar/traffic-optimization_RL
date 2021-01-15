import torch
import numpy as np
import traci


class TLEnv():
    def __init__(self, tl_list):
        self.tl_list = tl_list
        return state

    def get_state(self, action):
        for _, tl in enumerate(self.tl_list):
            lane_list = traci.trafficlight.getControlledLanes(tl)

        return state

    def get_reward(self, action):

        return reward

import torch
import numpy as np
import traci


class TLEnv():
    def __init__(self, tl_rlList, configs):
        self.configs = configs
        self.tl_rlList = tl_rlList
        self.tl_list = traci.trafficlight.getIDList()
        self.phase_size = len(traci.trafficlight.getPhase(self.tl_list[0]))
        for _, rl_tl in enumerate(self.tl_list):
        self._toState(traci)
        return state

    def get_state(self):
        phase = list()
        state = torch.Tensor()
        for _, tl_rl in enumerate(self.tl_rlList):
            phase.append(traci.trafficlight.getPhase(tl_rl))
        for _, p in enumerate(phase):
            torch.cat(state, self._toState(p))
        return state

    def step(self, action):
        phase = self._toPhase(action)  # action을 분해
        for _, tl_rl in enumerate(self.tl_rlList):
            traci.trafficlight.setRedYellowGreenState()

    def get_reward(self):

        return reward

    def _toPhase(self, action):  # action을 해석가능한 phase로 변환
        '''
        right: green signal
        straight: green=1, yellow=x, red=0 <- x is for changing
        left: green=1, yellow=x, red=0 <- x is for changing
        '''
        signal = list()
        phase = str()
        for _, a in enumerate(action):
            signal.append(self._getMovement(a))
        for i in range(4):  # 4차로
            phase = phase + 'g'+self.configs['numLane']*signal[2*i] + \
                signal[2*i+1]+'r'  # 마지막 r은 u-turn
        print(phase)
        return phase

    def _toState(self, phase):  # env의 phase를 해석불가능한 state로 변환
        for i in range(4):  # 4차로

        return state

    def _getMovement(self, num):
        if num == 1:
            return 'G'
        elif num == 0:
            return 'r'
        else:
            return 'y'

    def _mappingMovement(self, movement):
        if movement == 'r':
            return 0
        elif movement == 'G':
            return 1
        else:
            return 'y'

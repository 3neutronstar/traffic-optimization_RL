import torch
import numpy as np
import traci
from Env.base import baseEnv


class TLEnv(baseEnv):
    def __init__(self, tl_rlList, optimizer, configs):
        self.configs = configs
        self.tl_rlList = tl_rlList
        self.optimizer = optimizer
        self.tl_list = traci.trafficlight.getIDList()
        self.edge_list = traci.edge.getIDList()
        self.interest_inEdge=dict()
        self.interest_outEdge=
        for _, edges in enumerate(self.edge_list):
            for j, rl_node in enumerate(self.tl_rlList):
                if edges[-5:]==rl_node: #outflow
                    self.interest_outEdge.append(edges)
                if edges[:5]==rl_node: # inflow
                    self.interest_inEdge.append(edges)
        self.phase_size = len(traci.trafficlight.getPhase(self.tl_list[0]))


    def get_state(self):
        phase = list()
        state = torch.Tensor(device=self.configs['device'],dtype=torch.int)
        for _, edge in enumerate(edge_list): # 이 부분을 밖에서 list로 구성해오면 쉬움
            if edge[-5:]=='n_2_2': # outflow 여기에 n_2_2대신에 tl_id를 넣으면 pressure가 되는 것
                inflow+=traci.edge.getLastStepVehicleNumber(edge)
            elif edge[:5]=='n_2_2': # inflow
                outflow+=traci.edge.getLastStepVehicleNumber(edge)
        phase=traci.trafficlight.getRedYellowGreenState('n_2_2')
        for _, tl_rl in enumerate(self.tl_rlList):
            phase.append(traci.trafficlight.getPhase(tl_rl))
        for _, p in enumerate(phase):
            state = torch.cat(state, self._toState(p))
        return state

    def step(self, action):
        phase = self._toPhase(action)  # action을 분해
        for _, tl_rl in enumerate(self.tl_rlList):
            traci.trafficlight.setRedYellowGreenState(tl_rl, phase)

    def get_reward(self):
        '''
        reward function
        Max Pressure based control
        '''

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
        state = torch.zeros(8, dtype=torch.int)
        for i in range(4):  # 4차로
            phase = phase[1:]  # 우회전
            state[i] = self._mappingMovement(phase[0])  # 직진신호 추출
            phase = phase[3:]  # 직전
            state[i+1] = self._mappingMovement(phase[0])  # 좌회전신호 추출
            phase = phase[1:]  # 좌회전
            phase = phase[1:]  # 유턴

        return state

    def _getMovement(self, num):
        if num == 1:
            return 'G'
        elif num == 0:
            return 'r'
        else:
            return 'y'

    def _mappingMovement(self, movement):
        if movement == 'G':
            return 1
        elif movement == 'r':
            return 0
        else:
            return -1  # error

import torch
import numpy as np
import traci
from Env.base import baseEnv
from copy import deepcopy


class TLEnv(baseEnv):
    def __init__(self, tl_rlList, configs):
        self.configs = configs
        self.tl_rlList = tl_rlList
        self.tl_list = traci.trafficlight.getIDList()
        self.edge_list = traci.edge.getIDList()
        self.pressure = 0
        '''
        up right down left 순서대로 저장

        '''
        # pressure_dict=dict()
        # for i,edge in enumerate(self.edge_list):
        #     if edge[4]=='u': #
        #         string=
        self.interest_list = [
            {
                'id': 'u_1_1',
                'inflow': 'n_1_0_to_n_1_1',
                'outflow': 'n_1_1_to_n_1_2',
            },
            {
                'id': 'r_1_1',
                'inflow': 'n_2_1_to_n_1_1',
                'outflow': 'n_1_1_to_n_0_1',
            },
            {
                'id': 'd_1_1',
                'inflow': 'n_1_2_to_n_1_1',
                'outflow': 'n_1_1_to_n_1_0',
            },
            {
                'id': 'l_1_1',
                'inflow': 'n_0_1_to_n_1_1',
                'outflow': 'n_1_1_to_n_2_1',
            }
        ]

        # for _, edges in enumerate(self.edge_list):
        #     for j, rl_node in enumerate(self.tl_rlList):
        #         if edges[-5:]==rl_node: #outflow
        #             self.interest_outEdge.append(edges)
        #         if edges[:5]==rl_node: # inflow
        #             self.interest_inEdge.append(edges)
        self.phase_size = len(
            traci.trafficlight.getRedYellowGreenState(self.tl_list[0]))

    def get_state(self):
        phase = list()
        state = torch.zeros(
            (2, 8), device=self.configs['device'], dtype=torch.int)  # 기준
        vehicle_state = torch.zeros(
            (8, 1), device=self.configs['device'], dtype=torch.int)
        # for _, edge in enumerate(edge_list): # 이 부분을 밖에서 list로 구성해오면 쉬움
        #     if edge[-5:]=='n_2_2': # outflow 여기에 n_2_2대신에 tl_id를 넣으면 pressure가 되는 것
        #         inflow+=traci.edge.getLastStepVehicleNumber(edge)
        #     elif edge[:5]=='n_2_2': # inflow
        #         outflow+=traci.edge.getLastStepVehicleNumber(edge)

        # 변환
        for _, tl_rl in enumerate(self.tl_rlList):
            phase.append(traci.trafficlight.getRedYellowGreenState(tl_rl))

        # 1교차로용 n교차로는 추가요망
        phase_state = self._toState(phase[0])
        for i, interest in enumerate(self.interest_list):
            # 죄회전용 추가 필요
            vehicle_state[2*i +
                          1] = traci.edge.getLastStepVehicleNumber(interest['inflow'])
        vehicle_state = torch.transpose(vehicle_state, 0, 1)
        state = torch.cat((vehicle_state, phase_state), dim=0)
        return state

    def collect_state(self):
        '''
        갱신 및 점수 확정용 함수
        '''
        inflow_rate = 0
        outflow_rate = 0
        for _, interest in enumerate(self.interest_list):
            inflow_rate += traci.edge.getLastStepVehicleNumber(
                interest['inflow'])
            outflow_rate += traci.edge.getLastStepVehicleNumber(
                interest['outflow'])
        self.pressure += (outflow_rate-inflow_rate)

    def step(self, action):
        '''
        agent 의 action 적용 및 reward 계산
        '''
        phase = self._toPhase(action)  # action을 분해

        # action을 environment에 등록 후 상황 살피기
        for _, tl_rl in enumerate(self.tl_rlList):
            traci.trafficlight.setRedYellowGreenState(tl_rl, phase)

        # reward calculation and save

    def get_reward(self):
        '''
        reward function
        Max Pressure based control
        각 node에 대해서 inflow 차량 수와 outflow 차량수 + 해당 방향이라는 전제에서
        '''
        reward = deepcopy(self.pressure)
        self.pressure = 0
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
            phase = phase + 'g'+self.configs['num_lanes']*signal[2*i] + \
                signal[2*i+1]+'r'  # 마지막 r은 u-turn
        return phase

    def _toState(self, phase):  # env의 phase를 해석불가능한 state로 변환
        state = torch.zeros(
            (8, 1), device=self.configs['device'], dtype=torch.int16)
        for i in range(4):  # 4차로
            phase = phase[:][1:]  # 우회전
            state[i] = self._mappingMovement(phase[0])  # 직진신호 추출
            phase = phase[3:]  # 직전
            state[i+1] = self._mappingMovement(phase[0])  # 좌회전신호 추출
            phase = phase[1:]  # 좌회전
            phase = phase[1:]  # 유턴
        state = torch.transpose(state, 0, 1)
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

import torch
import numpy as np
import traci
from Env.base import baseEnv
from copy import deepcopy


class TL1x1Env(baseEnv):
    def __init__(self, tl_rl_list, configs):
        self.configs = configs
        self.tl_rl_list = tl_rl_list
        self.tl_list = traci.trafficlight.getIDList()
        self.edge_list = traci.edge.getIDList()
        self.pressure = 0
        '''
        up right down left 순서대로 저장

        '''
        # grid_num 3일 때
        # self.interest_list = [
        #     {
        #         'id': 'u_1_1',
        #         'inflow': 'n_1_0_to_n_1_1',
        #         'outflow': 'n_1_1_to_n_1_2',
        #     },
        #     {
        #         'id': 'r_1_1',
        #         'inflow': 'n_2_1_to_n_1_1',
        #         'outflow': 'n_1_1_to_n_0_1',
        #     },
        #     {
        #         'id': 'd_1_1',
        #         'inflow': 'n_1_2_to_n_1_1',
        #         'outflow': 'n_1_1_to_n_1_0',
        #     },
        #     {
        #         'id': 'l_1_1',
        #         'inflow': 'n_0_1_to_n_1_1',
        #         'outflow': 'n_1_1_to_n_2_1',
        #     }
        # ]

        # # grid_num 1일때
        self.interest_list = [
            {
                'id': 'u_0_0',
                'inflow': 'n_0_u_to_n_0_0',
                'outflow': 'n_0_0_to_n_0_d',
            },
            {
                'id': 'r_0_0',
                'inflow': 'n_0_r_to_n_0_0',
                'outflow': 'n_0_0_to_n_0_l',
            },
            {
                'id': 'd_0_0',
                'inflow': 'n_0_d_to_n_0_0',
                'outflow': 'n_0_0_to_n_0_u',
            },
            {
                'id': 'l_0_0',
                'inflow': 'n_0_l_to_n_0_0',
                'outflow': 'n_0_0_to_n_0_r',
            }
        ]

        self.phase_size = len(
            traci.trafficlight.getRedYellowGreenState(self.tl_list[0]))

    def get_state(self):
        phase = list()
        state = torch.zeros(
            (1, self.configs['state_space']), device=self.configs['device'], dtype=torch.int)  # 기준
        vehicle_state = torch.zeros(
            (int(self.configs['state_space']-8), 1), device=self.configs['device'], dtype=torch.int)  # -8은 phase크기
        # 변환
        for _, tl_rl in enumerate(self.tl_rl_list):
            phase.append(traci.trafficlight.getRedYellowGreenState(tl_rl))

        # 1교차로용 n교차로는 추가요망
        phase_state = self._toState(phase[0])/25.0
        for i, interest in enumerate(self.interest_list):
            # 죄회전용 추가 필요
            vehicle_state[i] = traci.edge.getLastStepVehicleNumber(
                interest['inflow'])
        vehicle_state = torch.transpose(vehicle_state, 0, 1)
        state = torch.cat((vehicle_state, phase_state),
                          dim=1)  # 여기 바꿨다 문제 생기면 여기임 암튼 그럼

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
        for _, tl_rl in enumerate(self.tl_rl_list):
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
            (8, 1), device=self.configs['device'], dtype=torch.int)
        for i in range(4):  # 4차로
            phase = phase[:][1:]  # 우회전
            state[i] = self._mappingMovement(phase[0])  # 직진신호 추출
            phase = phase[self.configs['num_lanes']:]  # 직전
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

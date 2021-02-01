import torch
import numpy as np
import traci
from Env.base import baseEnv
from copy import deepcopy

class TL3x3Env(baseEnv):
    def __init__(self, configs):
        self.configs = configs
        self.tl_rl_list = self.configs['tl_rl_list']
        self.tl_list = traci.trafficlight.getIDList()
        self.edge_list = traci.edge.getIDList()
        self.pressure = 0
        self.reward = 0
        self.phase_list = self._phase_list()
        self.left_lane_num = self.configs['num_lanes']-1
        self.num_vehicles_list=list()
        self.inflow=list()
        '''
        up right down left 순서대로 저장

        '''
        # grid_num 3일 때
        if self.configs['grid_num']==3:
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

        self.phase_size = len(
            traci.trafficlight.getRedYellowGreenState(self.tl_list[0]))

    def get_state(self):
        '''
        합쳐서 return
        '''
        state=torch.cat(self.num_vehicles_list,1) # 1차에 대해 합
        self.num_vehicles_list=list()
        return state

    def collect_state(self,phase_number):
        '''
        갱신 및 점수 확정용 함수
        '''
        vehicle_state = torch.zeros(
            (self.configs['state_space'], 1), device=self.configs['device'], dtype=torch.float)  # -8은 phase크기
        # vehicle state
        for i,interest in enumerate(self.interest_list): # 4개에 대해서 수집
            left_movement = traci.lane.getLastStepHaltingNumber(
                interest['inflow']+'_{}'.format(self.left_lane_num))
            # 직진
            vehicle_state[i*2] = traci.edge.getLastStepHaltingNumber(
                interest['inflow'])# -left_movement  # 가장 좌측에 멈춘 친구를 왼쪽차선 이용자로 판단
            # 좌회전
            vehicle_state[i*2+1] = left_movement
            self.inflow.append(vehicle_state)# for reward 연산량 감소
        
        self.num_vehicles_list[phase_number]=vehicle_state

    def step(self, action):
        '''
        agent 의 action 적용 및 reward 계산
        '''
        self.reward=0
        phases_length = self._toPhaseLength(action)  # action을 분해
        for i,phase in enumerate(self.phase_list):
            traci.trafficlight.setRedYellowGreenState(self.tl_rl_list[0],phase)
            for _ in range(phases_length[i]):
                traci.simulationStep()
                step+=1
            self.collect_state(phase_number=i)
            self.reward+=self.get_reward()
        next_state=self.get_state()
        reward=self.reward 
        return next_state,reward

        # reward calculation and save

    def get_reward(self):
        '''
        reward function
        Max Pressure based control
        각 node에 대해서 inflow 차량 수와 outflow 차량수 + 해당 방향이라는 전제에서
        '''
        outflow_rate = 0
        inflow_rate=0
        for _, interest in enumerate(self.interest_list):
            outflow_rate += traci.edge.getLastStepVehicleNumber(
                interest['outflow'])
        for inflow in self.inflow:
            inflow_rate+=inflow
        self.pressure = (inflow_rate-outflow_rate)
        reward = - self.pressure
        self.pressure = 0
        self.inflow=list() #초기화
        return reward

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('episode/reward', self.reward,
                          self.configs['max_steps']*epoch)  # 1 epoch마다
        # clear the value once in an epoch
        self.reward = 0 
    
    def _toSplit(self,action):
        matrix_actions=[[0,0,0,0],[1,0,0,-1],[1,0,-1,0],[1,-1,0,0],[0,1,0,-1],[0,1,-1,0],[0,0,1,-1],
        [1,0,0,-1],[1,0,-1,0],[1,0,0,-1],[0,1,0,-1],[0,1,-1,0],[0,0,1,-1]]
        return matrix_actions[action]

    def _toPhaseLength(self,action):
        phases=self.configs['min_phase']+self._toSplit(action[0])*action[1]
        return phases

    def _phase_list(self):
        num_lanes = self.configs['num_lanes']
        g = 'G'
        r = 'r'
        if self.configs['num_phase']==4:
            phase_list = [
                'r{2}{1}gr{2}{3}rr{2}{1}gr{2}{3}r'.format( # 위좌아래좌
                    g*num_lanes, g, r*num_lanes, r),
                'G{0}{3}rr{2}{3}rG{0}{3}rr{2}{3}r'.format( #위직아래직
                    g*num_lanes, g, r*num_lanes, r),  # current
                'r{2}{3}rr{2}{1}gr{2}{3}rr{2}{1}g'.format( #좌좌우좌
                    g*num_lanes, g, r*num_lanes, r),
                'r{2}{3}rG{0}{3}rr{2}{3}rG{0}{3}g'.format( #좌직우직
                    g*num_lanes, g, r*num_lanes, r),  # current
            ]
        return phase_list
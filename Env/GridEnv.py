import torch
import numpy as np
import traci
from Env.base import baseEnv
from copy import deepcopy


class GridEnv(baseEnv):
    def __init__(self, configs):
        super().__init__(configs)
        self.configs = configs
        self.tl_list = traci.trafficlight.getIDList()
        self.tl_rl_list = self.configs['tl_rl_list']
        self.side_list = ['u', 'r', 'd', 'l']
        self.interest_list = self._generate_interest_list()
        self.phase_size = len(
            traci.trafficlight.getRedYellowGreenState(self.tl_list[0]))
        self.pressure = 0
        self.reward = 0
        self.state_space = self.configs['state_space']
        self.vehicle_state_space = 8
        self.phase_state_space = 8
        self.action_size = len(self.tl_rl_list)
        self.left_lane_num = self.configs['num_lanes']-1
        self.node_interest_pair = dict()
        self.phase_list = self._phase_list()
        for _, node in enumerate(self.configs['node_info']):
            if node['id'][-1] not in self.side_list:
                self.node_interest_pair['{}'.format(
                    node['id'])] = list()
                for _, interest in enumerate(self.interest_list):
                    if node['id'][-3:] == interest['id'][-3:]:  # 좌표만 받기
                        self.node_interest_pair['{}'.format(
                            node['id'])].append(interest)

    def _generate_interest_list(self):
        interest_list = list()
        node_list = self.configs['node_info']
        x_y_end = self.configs['grid_num']-1
        for _, node in enumerate(node_list):
            if node['id'][-1] not in self.side_list:
                x = int(node['id'][-3])
                y = int(node['id'][-1])
                left_x = x-1
                left_y = y
                right_x = x+1
                right_y = y
                down_x = x
                down_y = y+1  # 아래로가면 y는 숫자가 늘어남
                up_x = x
                up_y = y-1  # 위로가면 y는 숫자가 줄어듦

                if x == 0:
                    left_y = 'l'
                    left_x = y
                if y == 0:
                    up_y = 'u'
                if x == x_y_end:
                    right_y = 'r'
                    right_x = y
                if y == x_y_end:
                    down_y = 'd'
                # up
                interest_list.append(
                    {
                        'id': 'u_{}'.format(node['id'][2:]),
                        'inflow': 'n_{}_{}_to_n_{}_{}'.format(up_x, up_y, x, y),
                        'outflow': 'n_{}_{}_to_n_{}_{}'.format(x, y, up_x, up_y),
                    }
                )
                # right
                interest_list.append(
                    {
                        'id': 'r_{}'.format(node['id'][2:]),
                        'inflow': 'n_{}_{}_to_n_{}_{}'.format(right_x, right_y, x, y),
                        'outflow': 'n_{}_{}_to_n_{}_{}'.format(x, y, right_x, right_y),
                    }
                )
                # down
                interest_list.append(
                    {
                        'id': 'd_{}'.format(node['id'][2:]),
                        'inflow': 'n_{}_{}_to_n_{}_{}'.format(down_x, down_y, x, y),
                        'outflow': 'n_{}_{}_to_n_{}_{}'.format(x, y, down_x, down_y),
                    }
                )
                # left
                interest_list.append(
                    {
                        'id': 'l_{}'.format(node['id'][2:]),
                        'inflow': 'n_{}_{}_to_n_{}_{}'.format(left_x, left_y, x, y),
                        'outflow': 'n_{}_{}_to_n_{}_{}'.format(x, y, left_x, left_y),
                    }
                )

        return interest_list

    def get_state(self):
        state_set = tuple()
        phase = list()
        for i, tl_rl in enumerate(self.tl_rl_list):
            state = torch.zeros(  # 1개 단위로 만들어서 붙임
                (1, self.configs['state_space']), device=self.configs['device'], dtype=torch.int)  # 기준
            vehicle_state = torch.zeros(
                (self.vehicle_state_space, 1), device=self.configs['device'], dtype=torch.int)
            phase = traci.trafficlight.getRedYellowGreenState(tl_rl)
            # n교차로
            phase_state = self._toState(phase).view(
                1, -1).to(self.configs['device'])

            # vehicle state
            for interest in self.node_interest_pair:
                for j, pair in enumerate(self.node_interest_pair[interest]):
                    left_movement = traci.lane.getLastStepHaltingNumber(
                        pair['inflow']+'_{}'.format(self.left_lane_num))  # 멈춘애들 계산
                    # 직진
                    vehicle_state[j*2] = traci.edge.getLastStepHaltingNumber(
                        pair['inflow'])-left_movement  # 가장 좌측에 멈춘 친구를 왼쪽차선 이용자로 판단
                    # 좌회전
                    vehicle_state[j*2+1] = left_movement
            vehicle_state = torch.transpose(vehicle_state, 0, 1)

            state = torch.cat((vehicle_state, phase_state),  # vehicle
                              dim=1).view(1, 1, -1)  # 여기 바꿨다 문제 생기면 여기임 암튼 그럼

            state_set += tuple(state)
        state_set = torch.cat(state_set, dim=1).float()

        return state_set

    def collect_state(self):
        '''
        갱신 및 점수 확정용 함수
        '''
        inflow_rate = 0
        outflow_rate = 0
        for _, interest in enumerate(self.interest_list):
            inflow_rate += traci.edge.getLastStepHaltingNumber(
                interest['inflow'])
            outflow_rate += traci.edge.getLastStepVehicleNumber(
                interest['outflow'])
        self.pressure += (outflow_rate-inflow_rate)

    def step(self, action):
        '''
        agent 의 action 적용 및 reward 계산
        '''
        action = action.view(-1, self.action_size)
        # action을 environment에 등록 후 상황 살피기
        for i, tl_rl in enumerate(self.tl_rl_list):
            phase = self._toPhase(action[i])  # action을 분해
            traci.trafficlight.setRedYellowGreenState(tl_rl, phase)

        # reward calculation and save

    def get_reward(self):
        '''
        reward function
        Max Pressure based control
        각 node에 대해서 inflow 차량 수와 outflow 차량수 + 해당 방향이라는 전제에서
        '''
        self.reward += self.pressure
        self.pressure = 0
        return self.reward

    def update_tensorboard(self, writer, epoch):
        writer.add_scalar('episode/reward', self.reward,
                          self.configs['max_steps']*epoch)  # 1 epoch마다
        # clear the value once in an epoch
        self.reward = 0

    def _toPhase(self, action):  # action을 해석가능한 phase로 변환
        '''
        right: green signal
        straight: green=1, yellow=x, red=0 <- x is for changing
        left: green=1, yellow=x, red=0 <- x is for changing
        '''
        print(action)
        return self.phase_list[action]

    def _toState(self, phase):  # env의 phase를 해석불가능한 state로 변환
        state = torch.zeros(self.phase_state_space, dtype=torch.int)
        for i in range(4):  # 4차로
            phase = phase[1:]  # 우회전
            state[i] = self._mappingMovement(phase[0])  # 직진신호 추출
            phase = phase[self.configs['num_lanes']-1:]  # 직전
            state[i+1] = self._mappingMovement(phase[0])  # 좌회전신호 추출
            phase = phase[1:]  # 좌회전
            phase = phase[1:]  # 유턴
        state = state.view(1, -1).float()
        return state

    def _getMovement(self, num):
        if num == 1:
            return 'G'
        elif num == 0:
            return 'r'
        else:
            return 'y'

    def _mappingMovement(self, movement):
        if movement == 'G' or movement == 'g':
            return 1
        elif movement == 'r' or movement == 'R':
            return 0
        else:
            return -1  # error

    def _phase_list(self):
        num_lanes = self.configs['num_lanes']
        g = 'G'
        r = 'r'
        phase_list = [
            'G{0}{1}gr{2}{3}rr{2}{3}rr{2}{3}r'.format(
                g*num_lanes, g, r*num_lanes, r),
            'r{2}{1}gr{2}{3}rr{2}{1}gr{2}{3}r'.format(
                g*num_lanes, g, r*num_lanes, r),
            'r{2}{3}rr{2}{3}rG{0}{1}gr{2}{3}r'.format(
                g*num_lanes, g, r*num_lanes, r),
            'G{0}{3}rr{2}{3}rG{0}{3}rr{2}{3}r'.format(
                g*num_lanes, g, r*num_lanes, r),  # current
            'r{2}{3}rG{0}{1}gr{2}{3}rr{2}{3}r'.format(
                g*num_lanes, g, r*num_lanes, r),
            'r{2}{3}rr{2}{3}rr{2}{3}rG{0}{1}g'.format(
                g*num_lanes, g, r*num_lanes, r),
            'r{2}{3}rr{2}{1}gr{2}{3}rr{2}{1}r'.format(
                g*num_lanes, g, r*num_lanes, r),
            'r{2}{3}rG{0}{3}rr{2}{3}rG{0}{3}g'.format(
                g*num_lanes, g, r*num_lanes, r),  # current
        ]
        return phase_list

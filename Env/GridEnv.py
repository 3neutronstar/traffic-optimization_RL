import torch
import numpy as np
import traci
from Env.base import baseEnv
from copy import deepcopy


class GridEnv(baseEnv):
    def __init__(self, configs):
        super().__init__(configs)
        self.configs = configs
        self.tl_list=traci.trafficlight.getIDList()
        self.tl_rl_list=list()
        self.interest_list=self._generate_interest_list(self.configs['grid_side'])
        self.phase_size=len(traci.trafficlight.getRedYellowGreenState(self.tl_list[0]))
        self.pressure=0

    

    def _generate_interest_list(self,grid_side):
        interest_list=list()
        for _,node in enumerate(self.tl_rl_list):
            x=int(node[2]) # n_x_y
            y=int(node[-1])
            if x!='0' and y!='0' and x!=self.configs['grid_num']-1 and y!=self.configs['grid_num']-1:
                # up
                self.tl_rl_list.append(node)
                interest_list.append(
                    {
                        'id':'u_{}'.format(node[-3:]),
                        'inflow':'n_{}_{}_to_n_{}_{}'.format(x,y-1,x,y),
                        'outflow':'n_{}_{}_to_n_{}_{}'.format(x,y,x,y+1),
                    }
                )
                # right
                interest_list.append(
                    {
                        'id':'u_{}'.format(node[-3:]),
                        'inflow':'n_{}_{}_to_n_{}_{}'.format(x+1,y,x,y),
                        'outflow':'n_{}_{}_to_n_{}_{}'.format(x,y,x-1,y),
                    }
                )
                # down
                interest_list.append(
                    {
                        'id':'u_{}'.format(node[-3:]),
                        'inflow':'n_{}_{}_to_n_{}_{}'.format(x,y+1,x,y),
                        'outflow':'n_{}_{}_to_n_{}_{}'.format(x,y,x,y-1),
                    }
                )
                # left
                interest_list.append(
                    {
                        'id':'u_{}'.format(node[-3:]),
                        'inflow':'n_{}_{}_to_n_{}_{}'.format(x-1,y,x,y),
                        'outflow':'n_{}_{}_to_n_{}_{}'.format(x,y,x+1,y),
                    }
                )
        if grid_side=='out':
            raise NotImplementedError
            # x=int(node[2]) # n_x_y
            # y=int(node[-1])
            # for _,node in enumerate(self.tl_rl_list):
            #     if x!='0' or y!='0' or x!=self.configs['grid_num']-1 or y!=self.configs['grid_num']-1:
            #         # up
            #         interest_list.append(
            #             {
            #                 'id':'u_{}'.format(node[-3:]),
            #                 'inflow':'n_{}_{}_to_n_{}_{}'.format(x,y-1,x,y),
            #                 'outflow':'n_{}_{}_to_n_{}_{}'.format(x,y,x,y+1),
            #             }

        return interest_list

    def get_state(self):
        state_set=tuple()
        phase = list()
        for _, tl_rl in enumerate(self.tl_rl_list):
            state = torch.zeros(
                (1, self.configs['state_space']), device=self.configs['device'], dtype=torch.int)  # 기준
            vehicle_state = torch.zeros(
                (int(self.configs['state_space']-8), 1), device=self.configs['device'], dtype=torch.int)
            phase.append(traci.trafficlight.getRedYellowGreenState(tl_rl))
            # 1교차로용 n교차로는 추가요망
            phase_state = self._toState(phase[0])

        # 변환
            for i, interest in enumerate(self.interest_list):
                # 죄회전용 추가 필요
                vehicle_state[i] = traci.edge.getLastStepVehicleNumber(interest['inflow'])
            vehicle_state = torch.transpose(vehicle_state, 0, 1)
            state = torch.cat((vehicle_state, phase_state), dim=1)#여기 바꿨다 문제 생기면 여기임 암튼 그럼
            state_set+=state
        return state_set

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
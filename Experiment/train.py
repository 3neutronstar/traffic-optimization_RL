import json
import os
import sys
import time
import traci
import traci.constants as tc
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from utils import update_tensorboard
from Agent.base import merge_dict


def dqn_train(configs, time_data, sumoCmd):
    # Environment Setting
    from Agent.dqn import Trainer
    if configs['model'] == 'base':
        from Env.Env import TL3x3Env
    elif configs['model'] == 'frap':
        from Env.FRAP import TL3x3Env
    # EXP_CONFIG Setting
    NUM_EPOCHS = configs['num_epochs']
    MAX_STEPS = configs['max_steps']
    tl_rl_list = configs['tl_rl_list']
    epoch = 0
    # init agent and tensorboard writer
    # agent setting
    agent = Trainer(configs)
    writer = SummaryWriter(os.path.join(
        configs['current_path'], 'training_data', time_data))
    # save hyper parameters
    agent.save_params(time_data)
    # init training
    while epoch < NUM_EPOCHS:
        # Epoch Start
        traci.start(sumoCmd)
        step = 0
        # Epoch Start setting
        env = TL3x3Env(configs)
        done = False
        total_reward = 0
        reward = 0
        arrived_vehicles = 0
        # state initialization
        action = torch.tensor([[0, 0]], dtype=torch.int,
                              device=configs['device'])
        state, _, _, _ = env.step(action, step)

        # Time Check
        a = time.time()
        while step < MAX_STEPS:

            action = agent.get_action(state)
            # environment에 적용
            next_state, reward, step, info = env.step(
                action, step)  # action 적용함수
            arrived_vehicles += info
            # 20초 지연된 보상
            agent.save_replay(state, action, reward, next_state)  # dqn
            agent.update(done)
            state = next_state
            total_reward += reward

        b = time.time()
        traci.close()
        print("time:", b-a)
        epoch += 1
        # update hyper parameter
        agent.update_hyperparams(epoch)  # lr and epsilon upate
        if epoch % agent.configs['target_update_period'] == 0:
            agent.target_update()  # dqn
        # once in an epoch update tensorboard
        update_tensorboard(writer, epoch, env, agent, arrived_vehicles)
        print('======== {} epoch/ return: {} arrived number:{}'.format(epoch,
                                                                       total_reward, arrived_vehicles))
        if epoch % 50 == 0:
            agent.save_weights(
                configs['file_name']+'_{}_{}'.format(time_data, epoch))

    writer.close()


def super_dqn_train(configs, time_data, sumoCmd):
    '''
    mask
    If some agents' time step are over their period, then mask True.
    Other's matrix element continue False.
    '''
    from Agent.super_dqn import Trainer
    if configs['model'] == 'base':
        from Env.MultiEnv import GridEnv

    # rl_list 설정
    side_list = ['u', 'r', 'd', 'l']
    tl_rl_list = list()
    for _, node in enumerate(configs['node_info']):
        if node['id'][-1] not in side_list:
            tl_rl_list.append(node['id'])
    configs['tl_rl_list'] = tl_rl_list
    num_agent = len(configs['tl_rl_list'])
    MAX_PHASES = 4  # 나중에 이거를 바꿔서 torch.max(전체 phase에서)
    NUM_EPOCHS = configs['num_epochs']
    MAX_STEPS = configs['max_steps']
    configs['rate_action_space'] = 13
    # time action space지정 (무조건 save param 이후 list화 시키고 나면 이전으로 옮길 것)
    # TODO 여기 홀수일 때, 어떻게 할 건지 지정해야함
    configs['time_action_space'] = (torch.min(torch.tensor(configs['max_phase'])-torch.tensor(
        configs['common_phase']), torch.tensor(configs['common_phase'])-torch.tensor(configs['min_phase']))/2).mean(dim=1).int().tolist()

    # init agent and tensorboard writer
    agent = Trainer(configs)
    writer = SummaryWriter(os.path.join(
        configs['current_path'], 'training_data', time_data))
    # save hyper parameters
    agent.save_params(time_data)
    # init training
    epoch = 0
    OFFSET = torch.tensor([0 for i in range(num_agent)],  # i*10
                          device=configs['device'], dtype=torch.float)
    MAX_PERIOD = torch.tensor([160 for i in range(
        num_agent)], device=configs['device'], dtype=torch.float)
    phase_num_matrix = torch.tensor(
        [len(phase) for i, phase in enumerate(configs['max_phase'])])
    while epoch < NUM_EPOCHS:
        step = 0
        traci.start(sumoCmd)
        env = GridEnv(configs)

        # Mask Matrix
        mask_matrix = torch.zeros(
            (num_agent), dtype=torch.bool, device=configs['device'])

        # MAX Period까지만 증가하는 t
        t_agent = torch.zeros(
            (num_agent), dtype=torch.int, device=configs['device'])

        # Action Matrix : 비교해서 동일할 때 collect_state, 없는 state는 zero padding
        action_matrix = torch.zeros(
            (num_agent, MAX_PHASES), device=configs['device'])  # 노란불 3초 해줘야됨
        action_index_matrix = torch.zeros(
            (num_agent), dtype=torch.long, device=configs['device'])  # 현재 몇번째 phase인지
        yellow_mask = torch.zeros(
            (num_agent), dtype=torch.bool, device=configs['device'])  # 현재 몇번째 phase인지

        # state initialization
        state = env.collect_state(mask_matrix, yellow_mask)
        total_reward = 0

        # agent setting
        arrived_vehicles = 0
        a = time.time()

        while step < MAX_STEPS:

            # action 을 정하고
            actions = agent.get_action(state, mask_matrix)
            action_matrix = env.calc_action(action_matrix,
                                            actions, mask_matrix)  # action형태로 변환 # 다음으로 넘어가야할 시점에 대한 matrix
            # 누적값으로 나타남

            # 전체 1초증가 # traci는 env.step에
            step+=1
            t_agent += 1
            # 최대에 도달하면 0으로 초기화 (offset과 비교)
            update_matrix = torch.eq(t_agent % MAX_PERIOD, OFFSET)
            t_agent[update_matrix] = 0
            # 넘어가야된다면 action index증가 (by tensor slicing+yellow signal)
            action_update_matrix = torch.eq(
                t_agent, action_matrix[0, action_index_matrix]).view(num_agent)  # 0,인 이유는 인덱싱

            action_index_matrix[action_update_matrix] += 1
            # agent의 최대 phase를 넘어가면 해당 agent의 action index 0으로 초기화
            # print(action_index_matrix)
            clear_matrix = torch.ge(action_index_matrix, phase_num_matrix)
            action_index_matrix[clear_matrix] = 0
            # mask update 요망 matrix True로 전환

            mask_matrix[clear_matrix] = True
            mask_matrix[~clear_matrix] = False

            # 만약 action이 끝나기 3초전이면 yellow signal 적용, reward 갱신

            yellow_mask = torch.eq(
                t_agent-3, action_matrix[0, action_index_matrix])
            for y in torch.nonzero(yellow_mask):
                traci.trafficlight.setRedYellowGreenState(
                    tl_rl_list[y], 'y'*20)
            # environment에 적용
            # action 적용함수, traci.simulationStep 있음
            next_state = env.step(actions, action_index_matrix, yellow_mask)

            # env속에 agent별 state를 꺼내옴, max_offset+period 이상일 때 시작
            if step >= int(torch.max(OFFSET)+torch.max(MAX_PERIOD)):
                rep_state, rep_action, rep_reward, rep_next_state = env.get_state(
                    mask_matrix)
                agent.save_replay(rep_state, rep_action, rep_reward,
                                  rep_next_state, mask_matrix)  # dqn
                total_reward += rep_reward.sum()

            agent.update(mask_matrix)
            state = next_state
            # info
            arrived_vehicles += traci.simulation.getArrivedNumber()

        agent.update_hyperparams(epoch)  # lr and epsilon upate
        if epoch % agent.configs['target_update_period'] == 0:
            agent.target_update()  # dqn
        b = time.time()
        traci.close()
        print("time:", b-a)
        epoch += 1
        # once in an epoch
        update_tensorboard(writer, epoch, env, agent, arrived_vehicles)
        print('======== {} epoch/ return: {} arrived number:{}'.format(epoch,
                                                                       total_reward.sum(), arrived_vehicles))
        if epoch % 50 == 0:
            agent.save_weights(
                configs['file_name']+'_{}_{}'.format(time_data, epoch))

    writer.close()

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
        action_distribution = tuple()
        # Epoch Start setting
        env = TL3x3Env(configs)
        traci.trafficlight.setRedYellowGreenState(tl_rl_list[0], 'G{0}{3}rr{2}{3}rG{0}{3}rr{2}{3}r'.format(
            'G'*configs['num_lanes'], 'G', 'r'*configs['num_lanes'], 'r'))
        done = False
        total_reward = 0
        reward = 0
        arrived_vehicles = 0
        # state initialization
        state = env.get_state()
        # Time Check
        a = time.time()
        while step < MAX_STEPS:

            # action 을 받음
            actions = agent.get_action(state)
            action_distribution += tuple(actions.unsqueeze(1))
            # n차 phase 돌리기
            for i,phase in enumerate(phases):
                traci.simulationStep()
                step+=actions[i]
                traci.trafficlight.setRedYellowGreenState(tl_rl_list[0], 'y'*20)
                
            for _ in range(num_step):  # 10초마다 행동 갱신
                traci.simulationStep()
                env.collect_state()
                arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput

            # environment에 적용
            env.step(action)  # action 적용함수

            #적용 후 20초 진행
            arrived_vehicles += simulation_step(env, 20)
            step+=20
            next_state = env.get_state()  # 다음스테이트

            reward = env.get_reward()  # 20초 지연된 보상
            agent.save_replay(state, action, reward, next_state)  # dqn
            agent.update(done)
            state = next_state
            total_reward += reward
            before_action=action
            # 20초 끝나고 yellow 4초

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
    from Agent.super_dqn import Trainer
    if configs['model'] == 'base':
        from Env.MultiEnv import GridEnv
    # elif configs['model'] == 'frap':
    #     from Env.FRAP import TL3x3Env # will be added
    side_list = ['u', 'r', 'd', 'l']
    tl_rl_list = list()
    for _, node in enumerate(configs['node_info']):
        if node['id'][-1] not in side_list:
            tl_rl_list.append(node['id'])
    configs['tl_rl_list'] = tl_rl_list
    NUM_EPOCHS = configs['num_epochs']
    MAX_STEPS = configs['max_steps']
    # init agent and tensorboard writer
    agent = Trainer(configs)
    writer = SummaryWriter(os.path.join(
        configs['current_path'], 'training_data', time_data))
    # save hyper parameters
    agent.save_params(time_data)
    # init training
    epoch = 0
    while epoch < NUM_EPOCHS:
        traci.start(sumoCmd)
        for tl_rl in tl_rl_list:
            traci.trafficlight.setRedYellowGreenState(tl_rl, 'G{0}{3}rr{2}{3}rG{0}{3}rr{2}{3}r'.format(
                'G'*configs['num_lanes'], 'G', 'r'*configs['num_lanes'], 'r'))
        
        before_action= torch.ones((1,len(tl_rl_list),1),dtype=torch.int)
        env = GridEnv(configs)
        step = 0
        done = False
        # state initialization
        # agent setting
        total_reward = 0
        reward = 0
        arrived_vehicles = 0
        state = env.get_state()
        action_distribution = tuple()
        a = time.time()
        
        while step < MAX_STEPS:

            action = agent.get_action(state)
            action_distribution += tuple(action.unsqueeze(1))
            # action 을 정하고

            # action이 before_actio과 같으면 yellow없이 진행하고
            for tl_rl in tl_rl_list:
                idx=tl_rl_list.index(tl_rl)
                if before_action[0][idx]==action[0][idx]:
                    traci.trafficlight.setRedYellowGreenState(
                        tl_rl, 'y'*(3+configs['num_lanes'])*4)
            arrived_vehicles += simulation_step(env, 5)
            step+=5
            
            # environment에 적용
            env.step(action)  # action 적용함수

            #적용 후 20초 진행
            arrived_vehicles += simulation_step(env, 20)
            step+=20
            next_state = env.get_state()  # 다음스테이트

            reward = env.get_reward()  # 20초 지연된 보상
            agent.save_replay(state, action, reward, next_state)  # dqn
            agent.update(done)
            state = next_state
            total_reward += reward
            before_action=action
            # 20초 끝나고 yellow 4초

        agent.update_hyperparams(epoch)  # lr and epsilon upate
        if epoch % 2 == 0:
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

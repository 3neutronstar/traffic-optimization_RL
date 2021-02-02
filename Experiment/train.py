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
        step = 0
        traci.start(sumoCmd)
        env = GridEnv(configs)

        # Mask Matrix
        mask_matrix = torch.zeros(
            (1, len(configs['tl_rl_list'])), dtype=torch.bool, device=configs['device'])

        # state initialization
        state = env.get_state()
        total_reward = 0
        reward = 0

        # agent setting
        arrived_vehicles = 0
        a = time.time()

        while step < MAX_STEPS:

            # action 을 정하고
            action = agent.get_action(state, mask_matrix)

            # environment에 적용
            env.step(action, mask_matrix)  # action 적용함수

            next_state = env.get_state(mask_matrix)  # 다음스테이트

            reward = env.get_reward(mask_matrix)  # 20초 지연된 보상
            agent.save_replay(state, action, reward, next_state)  # dqn
            agent.update()
            state = next_state
            total_reward += reward
            # 20초 끝나고 yellow 4초

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

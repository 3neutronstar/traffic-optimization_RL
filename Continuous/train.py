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


def simulation_step(env, num_step):
    '''
    running_env
    num_step you want to step
    simulation step automatically
    '''
    arrived_vehicles = 0
    for _ in range(num_step):  # 10초마다 행동 갱신
        traci.simulationStep()
        env.collect_state()
        step += 1
        arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput
    return arrived_vehicles


def ddpg_train(configs, time_data, sumoCmd):
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

            action = agent.get_action(state)
            action_distribution += tuple(action.unsqueeze(1))
            env.step(action)  # action 적용함수

            arrived_vehicles += simulation_step(env, 20)
            next_state = env.get_state()  # 다음스테이트

            traci.trafficlight.setRedYellowGreenState(
                tl_rl_list[0], 'y'*28)

            arrived_vehicles += simulation_step(env, 5)

            reward = env.get_reward()  # 25초 지연된 보상
            agent.save_replay(state, action, reward, next_state)  # dqn
            agent.update(done)
            state = next_state
            total_reward += reward
        # update hyper parameter
        agent.update_hyperparams(epoch)  # lr and epsilon upate
        if epoch % agent.configs['target_update_period'] == 0:
            agent.target_update()  # dqn
        b = time.time()
        traci.close()
        print("time:", b-a)
        epoch += 1
        # once in an epoch update tensorboard
        update_tensorboard(writer, epoch, env, agent, arrived_vehicles)
        print('======== {} epoch/ return: {} arrived number:{}'.format(epoch,
                                                                       total_reward, arrived_vehicles))
        if epoch % 50 == 0:
            agent.save_weights(
                configs['file_name']+'_{}_{}'.format(time_data, epoch))

    writer.close()

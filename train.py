import json
import os
import sys
import time
import traci
import traci.constants as tc
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from utils import save_params, load_params, update_tensorboard
from gen_net import configs
from Agent.base import merge_dict


interest_list = [
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


def dqn_train(configs, time_data, sumoCmd):
    from Agent.dqn import Trainer
    if configs['model'] == 'base':
        from Env.env import TL3x3Env
    elif configs['model'] == 'frap':
        from Env.FRAP import TL3x3Env
    tl_rl_list = configs['tl_rl_list']
    NUM_EPOCHS = configs['num_epochs']
    MAX_STEPS = configs['max_steps']
    # init agent and tensorboard writer
    agent = Trainer(configs)
    writer = SummaryWriter(os.path.join(
        configs['current_path'], 'training_data', time_data))
    # save hyper parameters
    save_params(configs, time_data)
    # init training
    epoch = 0
    while epoch < NUM_EPOCHS:
        traci.start(sumoCmd)
        traci.trafficlight.setRedYellowGreenState(tl_rl_list[0], 'G{0}{1}gr{2}{3}rr{2}{3}rr{2}{3}r'.format(
            'G'*configs['num_lanes'], 'G', 'r'*configs['num_lanes'], 'r'))
        before_action = torch.tensor([[1]])  # 초기화
        env = TL3x3Env(tl_rl_list, configs)
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
            '''
            # state=env.get_state(action) #partial하게는 env에서 조정
            action=agent.get_action(state)
            env.step(action)
            reward=env.get_reward()
            next_state=env.get_state()
            # if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
            store transition in D (experience replay)
            Sample random minibatch from D
            step += 1
            state=next_state


            set yi
            '''

            action = agent.get_action(state)
            action_distribution += tuple(action.unsqueeze(1))
            env.step(action)  # action 적용함수

            for _ in range(20):  # 10초마다 행동 갱신
                traci.simulationStep()
                env.collect_state()
                step += 1
                arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput
            if before_action != action:
                traci.trafficlight.setRedYellowGreenState(
                    tl_rl_list[0], 'y'*28)

            for _ in range(5):  # 4번더
                traci.simulationStep()
                env.collect_state()
                step += 1
                arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput

            next_state = env.get_state()  # 다음스테이트
            reward = env.get_reward()  # 25초 지연된 보상
            agent.save_replay(state, action, reward, next_state)  # dqn
            agent.update(done)
            state = next_state
            total_reward += reward
            before_action = action

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
                                                                       total_reward, arrived_vehicles))
        if epoch % 50 == 0:
            agent.save_weights(
                configs['file_name']+'_{}_{}'.format(time_data, epoch))

    writer.close()


def REINFORCE_train(configs, time_data, sumoCmd):
    from Agent.REINFORCE import Trainer
    from Agent.REINFORCE import DEFAULT_CONFIG
    from Env.env import TL3x3Env
    tl_rl_list = configs['tl_rl_list']
    NUM_EPOCHS = configs['num_epochs']
    MAX_STEPS = configs['max_steps']
    # init agent and tensorboard writer
    agent = Trainer(configs)
    writer = SummaryWriter(os.path.join(
        configs['current_path'], 'training_data', time_data))
    # save hyper parameters
    save_params(merge_dict(configs, DEFAULT_CONFIG), time_data)
    # init training
    epoch = 0
    while epoch < NUM_EPOCHS:
        traci.start(sumoCmd)
        traci.trafficlight.setRedYellowGreenState(tl_rl_list[0], 'G{0}{1}gr{2}{3}rr{2}{3}rr{2}{3}r'.format(
            'G'*configs['num_lanes'], 'G', 'r'*configs['num_lanes'], 'r'))
        env = TL3x3Env(tl_rl_list, configs)
        # env = GridEnv( configs)
        step = 0
        done = False
        # state initialization
        # agent setting
        total_reward = 0
        reward = 0
        arrived_vehicles = 0
        state = env.get_state()
        while step < MAX_STEPS:

            action = agent.get_action(state)
            env.step(action)  # action 적용함수
            next_state = env.get_state()

            for _ in range(20):  # 10초마다 행동 갱신
                traci.simulationStep()
                env.collect_state()
                step += 1
                arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput
                '''
                응 뭐해야될지 알겠지? 모델링 다시해 ㅋㅋㅋㅋ step에다가 어지간한거 다 때려박자 ㅎ...

                '''
            # 20초 끝나고 yellow 4초
            traci.trafficlight.setRedYellowGreenState(tl_rl_list[0], 'y'*28)

            for _ in range(4):  # 4번더
                traci.simulationStep()
                env.collect_state()
                step += 1
                arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput

            traci.simulationStep()
            env.collect_state()
            step += 1
            arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput

            reward = env.get_reward()
            prob = agent.get_prob()
            agent.put_data((reward, prob[action]))

            state = next_state
            total_reward += reward
            if step > MAX_STEPS:
                done = True

        agent.update_hyperparams(epoch)  # lr and epsilon upate
        agent.update(done)
        traci.close()
        epoch += 1
        # once in an epoch
        update_tensorboard(writer, epoch, env, agent, arrived_vehicles)
        print('======== {} epoch/ return: {} arrived number:{}'.format(epoch,
                                                                       total_reward, arrived_vehicles))

    writer.close()


def a2c_train(configs, time_data, sumoCmd):
    from Agent.a2c import Trainer
    if configs['model'] == 'base':
        from Env.env import TL3x3Env
    elif configs['model'] == 'frap':
        from Env.FRAP import TL3x3Env
    tl_rl_list = configs['tl_rl_list']
    NUM_EPOCHS = configs['num_epochs']
    MAX_STEPS = configs['max_steps']
    # init agent and tensorboard writer
    agent = Trainer(configs)
    writer = SummaryWriter(os.path.join(
        configs['current_path'], 'training_data', time_data))
    # save hyper parameters
    save_params(configs, time_data)
    # init training
    epoch = 0
    while epoch < NUM_EPOCHS:
        traci.start(sumoCmd)
        traci.trafficlight.setRedYellowGreenState(tl_rl_list[0], 'G{0}{1}gr{2}{3}rr{2}{3}rr{2}{3}r'.format(
            'G'*configs['num_lanes'], 'G', 'r'*configs['num_lanes'], 'r'))
        before_action = torch.tensor([1])
        env = TL3x3Env(tl_rl_list, configs)
        # env = GridEnv( configs)
        step = 0
        # state initialization
        # agent setting
        total_reward = 0
        reward = 0
        arrived_vehicles = 0
        state = env.get_state()
        action_distribution = tuple()
        a = time.time()
        while step < MAX_STEPS:
            '''
            # state=env.get_state(action) #partial하게는 env에서 조정
            action=agent.get_action(state)
            env.step(action)
            reward=env.get_reward()
            next_state=env.get_state()
            # if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
            store transition in D (experience replay)
            Sample random minibatch from D
            step += 1
            state=next_state


            set yi
            '''

            action = agent.get_action(state)
            action_distribution += tuple(action.unsqueeze(1))
            env.step(action)  # action 적용함수
            next_state = env.get_state()  # 다음스테이트
            for _ in range(20):  # 10초마다 행동 갱신
                traci.simulationStep()
                env.collect_state()
                step += 1
                arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput

            if action != before_action:  # 다음 신호와 현재 신호가 다르면 yellow로 모두 변환
                traci.trafficlight.setRedYellowGreenState(
                    tl_rl_list[0], 'y'*28)

            for _ in range(4):  # 4번더
                traci.simulationStep()
                env.collect_state()
                step += 1
                arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput

            traci.simulationStep()
            env.collect_state()
            step += 1
            arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput

            reward = env.get_reward()  # 25초 지연된 보상
            agent.save_replay(state, action, reward, next_state)  # dqn
            agent.update()
            state = next_state
            total_reward += reward
            before_action = action

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
                                                                       total_reward, arrived_vehicles))
        if epoch % 50 == 0:
            agent.save_weights(
                configs['file_name']+'_{}_{}'.format(time_data, epoch))

    writer.close()


def ppo_train(configs, time_data, sumoCmd):
    from Agent.ppo import Trainer
    if configs['model'] == 'base':
        from Env.env import TL3x3Env
    elif configs['model'] == 'frap':
        from Env.FRAP import TL3x3Env
    tl_rl_list = configs['tl_rl_list']
    NUM_EPOCHS = configs['num_epochs']
    MAX_STEPS = configs['max_steps']
    # init agent and tensorboard writer
    agent = Trainer(configs)
    writer = SummaryWriter(os.path.join(
        configs['current_path'], 'training_data', time_data))
    # save hyper parameters
    save_params(configs, time_data)
    # init training
    epoch = 0
    while epoch < NUM_EPOCHS:
        traci.start(sumoCmd)
        traci.trafficlight.setRedYellowGreenState(tl_rl_list[0], 'G{0}{1}gr{2}{3}rr{2}{3}rr{2}{3}r'.format(
            'G'*configs['num_lanes'], 'G', 'r'*configs['num_lanes'], 'r'))
        env = TL3x3Env(tl_rl_list, configs)
        # env = GridEnv( configs)
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
            env.step(action)  # action 적용함수

            for _ in range(20):  # 10초마다 행동 갱신
                traci.simulationStep()
                env.collect_state()
                step += 1
                arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput
            next_state = env.get_state()  # 다음스테이트

            traci.trafficlight.setRedYellowGreenState(tl_rl_list[0], 'y'*28)

            for _ in range(4):  # 4번더
                traci.simulationStep()
                env.collect_state()
                step += 1
                arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput

            traci.simulationStep()
            env.collect_state()
            step += 1
            arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput

            reward = env.get_reward()  # 25초 지연된 보상
            agent.memory.rewards.append(reward)
            if step >= MAX_STEPS:
                done = True
            agent.memory.dones.append(done)
            state = next_state
            total_reward += reward

        if epoch % agent.configs['update_period'] == 0:
            agent.update()
        agent.update_hyperparams(epoch)  # lr update

        b = time.time()
        traci.close()
        print("time:", b-a)
        epoch += 1
        # once in an epoch
        update_tensorboard(writer, epoch, env, agent, arrived_vehicles)
        print('======== {} epoch/ return: {} arrived number:{}'.format(epoch,
                                                                       total_reward, arrived_vehicles))
        if epoch % 50 == 0:
            agent.save_weights(
                configs['file_name']+'_{}_{}'.format(time_data, epoch))

    writer.close()

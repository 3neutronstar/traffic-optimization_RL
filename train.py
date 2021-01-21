import json, os, sys, time
import traci
import traci.constants as tc
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from utils import save_params,load_params,update_tensorboard
from gen_net import configs

configs['input_size'] = 5*len(configs['tl_rl_list'])
configs['output_size'] = 8*len(configs['tl_rl_list'])
configs['action_space'] = 1*len(configs['tl_rl_list'])
configs['state_space'] = 5*len(configs['tl_rl_list'])

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

def dqn_train(configs,time_data,sumoCmd):
    from Env.env import TL3x3Env
    from Agent.dqn import Trainer
    from Env.GridEnv import GridEnv
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
        # for _ in range(250): # for stable learning
        #     traci.simulationStep()
        #     step += 1
        state = env.get_state()
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

            action = agent.get_action(state, reward)
            env.step(action)  # action 적용함수
            for _ in range(20):  # 10초마다 행동 갱신
                env.collect_state()
                arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput
                traci.simulationStep()
                step += 1

            env.collect_state()  # 1번더
            reward = env.get_reward()
            next_state = env.get_state()
            agent.save_replay(state, action, reward, next_state)  # dqn
            agent.update(done)

            state = next_state
            total_reward += reward
            if step > MAX_STEPS:
                done = True
            arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput

            step += 1
            traci.simulationStep()
            env.collect_state()

            # 20초 끝나고 yellow 4초
            traci.trafficlight.setRedYellowGreenState(tl_rl_list[0], 'y'*28)

            for _ in range(4):  # 4번더
                traci.simulationStep()
                arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput
                env.collect_state()
                step += 1

            if (epoch*MAX_STEPS+step) % 2000 == 0: # 총 몇번당 update
                if epoch%2==0:
                    agent.target_update()  # dqn
                agent.update_hyperparams(epoch) # lr and epsilon upate

        traci.close()
        epoch += 1
        # once in an epoch
        update_tensorboard(writer,epoch,env,agent,arrived_vehicles)
        print('======== {} epoch/ return: {} arrived number:{}'.format(epoch, total_reward, arrived_vehicles))
        if epoch % 50 == 0:
            agent.save_weights(
                configs['file_name']+'_{}_{}'.format(time_data, epoch))

    writer.close()


def RENINFORCE_train(configs,time_data,sumoCmd):
    from Env.env import TL3x3Env
    from Agent.REINFORCE import Trainer
    from Env.GridEnv import GridEnv
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
        while step < MAX_STEPS:

            action = agent.get_action(state, reward)
            env.step(action)  # action 적용함수

            for _ in range(20):  # 10초마다 행동 갱신
                env.collect_state()
                arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput
                traci.simulationStep()
                step += 1

            env.collect_state()  # 1번더
            reward = env.get_reward()
            next_state = env.get_state()
            agent.update(done)

            state = next_state
            total_reward += reward
            if step > MAX_STEPS:
                done = True
            arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput

            step += 1
            traci.simulationStep()
            env.collect_state()

            # 20초 끝나고 yellow 4초
            traci.trafficlight.setRedYellowGreenState(tl_rl_list[0], 'y'*28)

            for _ in range(4):  # 4번더
                traci.simulationStep()
                arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput
                env.collect_state()
                step += 1

            if (epoch*MAX_STEPS+step) % 2000 == 0: # 총 몇번당 update
                agent.update_hyperparams(epoch) # lr and epsilon upate

        traci.close()
        epoch += 1
        # once in an epoch
        update_tensorboard(writer,epoch,env,agent,arrived_vehicles)
        print('======== {} epoch/ return: {} arrived number:{}'.format(epoch, total_reward, arrived_vehicles))
        if epoch % 50 == 0:
            agent.save_weights(
                configs['file_name']+'_{}_{}'.format(time_data, epoch))

    writer.close()
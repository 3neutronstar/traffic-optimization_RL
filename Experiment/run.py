import argparse
import json
import os
import sys
import time
import torch
import torch.optim as optim
import traci
import random
import numpy as np
import traci.constants as tc
from sumolib import checkBinary
from utils import interest_list
from configs import EXP_CONFIGS, TRAFFIC_CONFIGS
from Agent.base import merge_dict


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="choose the mode",
        epilog="python run.py mode")

    # required input parameters
    parser.add_argument(
        'mode', type=str,
        help='train or test, simulate, "train_old" is the old version to train')
    parser.add_argument(
        '--network', type=str, default='grid',
        help='choose network in Env')
    # optional input parameters
    parser.add_argument(
        '--disp', type=str, default='no',
        help='show the process while in training')
    parser.add_argument(
        '--replay_name', type=str, default=None,
        help='activate only in test mode and write file_name to load weights.')
    parser.add_argument(
        '--algorithm', type=str, default='dqn',
        help='choose algorithm dqn, reinforce, a2c, ppo.')
    parser.add_argument(
        '--model', type=str, default='base',
        help='choose model base and FRAP.')
    parser.add_argument(
        '--gpu', type=bool, default=False,
        help='choose model base and FRAP.')
    return parser.parse_known_args(args)[0]


def train(flags, time_data, configs, sumoConfig):

    # check gui option
    if flags.disp == 'yes':
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    sumoCmd = [sumoBinary, "-c", sumoConfig, '--start']
    # configs setting
    configs['algorithm'] = flags.algorithm.lower()
    print("training algorithm: ", configs['algorithm'])
    configs['num_phase'] = 4
    if flags.algorithm.lower() == 'super_dqn':  # action space와 size 설정
        configs['action_space'] = configs['num_phase']
        configs['action_size'] = 1
        configs['state_space'] = 8  # 4phase에서 각각 받아오는게 아니라 마지막에 한번만 받음
        configs['model'] = 'base'
    elif flags.model.lower() == 'base':
        configs['action_space'] = 13
        configs['action_size'] = 2
        configs['state_space'] = 8
        configs['model'] = 'base'
        configs['time_size'] = int((torch.tensor(configs['phase_period'])
                                    - torch.tensor(configs['min_phase']).sum())/configs['num_phase'])  # 최대에서 최소 뺀 값이 size가 됨
        print(configs['time_size'])
    elif flags.model.lower() == 'frap':
        configs['action_space'] = configs['num_phase']
        configs['action_size'] = 1
        configs['state_space'] = 16
        configs['model'] = 'frap'

    if flags.algorithm.lower() == 'dqn':
        from train import dqn_train
        dqn_train(configs, time_data, sumoCmd)

    elif flags.algorithm.lower() == 'super_dqn':
        from train import super_dqn_train
        super_dqn_train(configs, time_data, sumoCmd)


def test(flags, configs, sumoConfig):
    from Env.Env import TL3x3Env
    from Agent.dqn import Trainer
    from Env.MultiEnv import GridEnv
    from utils import save_params, load_params, update_tensorboard

    # init test setting
    sumoBinary = checkBinary('sumo-gui')
    sumoCmd = [sumoBinary, "-c", sumoConfig]

    # setting the rl list
    tl_rl_list = configs['tl_rl_list']
    MAX_STEPS = configs['max_steps']
    reward = 0
    traci.start(sumoCmd)
    agent = Trainer(configs)
    # setting the replay
    if flags.replay_name is not None:
        agent.load_weights(flags.replay_name)
        configs = load_params(configs, flags.replay_name)
    env = TL3x3Env(configs)
    step = 0
    # state initialization
    state = env.get_state()
    # agent setting
    total_reward = 0
    arrived_vehicles = 0
    with torch.no_grad():
        while step < MAX_STEPS:

            action = agent.get_action(state)
            env.step(action)  # action 적용함수
            for _ in range(20):  # 10초마다 행동 갱신
                env.collect_state()
                traci.simulationStep()
                step += 1

            traci.trafficlight.setRedYellowGreenState(tl_rl_list[0], 'y'*28)
            for _ in range(5):
                traci.simulationStep()
                env.collect_state()
                step += 1

            reward = env.get_reward()
            next_state = env.get_state()
            # agent.save_replay(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            step += 1
            if step == MAX_STEPS:
                done = True
            # agent.update(done) # no update in
            # loss += agent.get_loss()  # 총 loss
            arrived_vehicles += traci.simulation.getArrivedNumber()  # throughput
            traci.simulationStep()
            # if step % 200 == 0:

        agent.target_update()
        traci.close()
        print('======== return: {} arrived number:{}'.format(
            total_reward, arrived_vehicles))


def simulate(flags, configs, sumoConfig):
    sumoBinary = checkBinary('sumo')
    sumoCmd = [sumoBinary, "-c", sumoConfig]
    MAX_STEPS = configs['max_steps']
    traci.start(sumoCmd)
    traci.simulation.subscribe([tc.VAR_ARRIVED_VEHICLES_NUMBER])
    traci.edge.subscribe('n_2_2_to_n_2_1', [
                         tc.LAST_STEP_VEHICLE_HALTING_NUMBER], 0, 2000)
    avg_waiting_time = 0
    avg_velocity = 0
    step = 0
    # agent setting
    arrived_vehicles = 0
    avg_velocity = 0
    while step < MAX_STEPS:

        traci.simulationStep()
        step += 1
        for _, edge in enumerate(interest_list):
            avg_waiting_time += traci.edge.getWaitingTime(edge['inflow'])

        # vehicle_list = traci.vehicle.getIDList()
        # for i, vehicle in enumerate(vehicle_list):
        #     speed = traci.vehicle.getSpeed(vehicle)
        #     avg_velocity = float((i)*avg_velocity+speed) / \
        #         float(i+1)  # incremental avg

        arrived_vehicles += traci.simulation.getAllSubscriptionResults()[
            ''][0x79]  # throughput

    traci.close()
    edgesss = traci.edge.getSubscriptionResults('n_2_2_to_n_2_1')
    print(edgesss)
    print('======== arrived number:{} avg waiting time:{},avg velocity:{}'.format(
        arrived_vehicles, avg_waiting_time/MAX_STEPS, avg_velocity))


def main(args):
    random_seed = 20000
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    flags = parse_args(args)
    configs = merge_dict(EXP_CONFIGS, TRAFFIC_CONFIGS)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda and flags.gpu == True else "cpu")
    # device = torch.device('cpu')
    print("Using device: {}".format(device))
    configs['device'] = str(device)
    configs['current_path'] = os.path.dirname(os.path.abspath(__file__))
    configs['mode'] = flags.mode.lower()
    # init train setting
    time_data = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))
    configs['time_data'] = str(time_data)

    # check the network
    if flags.network.lower() == 'grid':
        from grid import GridNetwork  # network바꿀때 이걸로 바꾸세요(수정 예정)
        configs['grid_num'] = 3
        if flags.algorithm.lower() == 'super_dqn':
            configs['grid_num'] = 3
        configs['file_name'] = '{}x{}grid'.format(
            configs['grid_num'], configs['grid_num'])
        network = GridNetwork(configs, grid_num=configs['grid_num'])
        network.generate_cfg(True, configs['mode'])
    # check the mode
    if configs['mode'] == 'train':
        configs['mode'] = 'train'
        sumoConfig = os.path.join(
            configs['current_path'], 'training_data', time_data, 'net_data', configs['file_name']+'_train.sumocfg')
        train(flags, time_data, configs, sumoConfig)
    elif configs['mode'] == 'test':
        configs['mode'] = 'test'
        sumoConfig = os.path.join(
            configs['current_path'], 'Net_data', configs['file_name']+'_test.sumocfg')
        test(flags, configs, sumoConfig)
    else:  # simulate
        configs['mode'] = 'simulate'
        sumoConfig = os.path.join(
            configs['current_path'], 'Net_data', configs['file_name']+'_simulate.sumocfg')
        simulate(flags, configs, sumoConfig)

    # check the environment
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")


if __name__ == '__main__':
    main(sys.argv[1:])

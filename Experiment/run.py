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
from configs import EXP_CONFIGS
from Agent.base import merge_dict,merge_dict_non_conflict


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
        help='choose network in Env or load from map file')
    # optional input parameters
    parser.add_argument(
        '--disp', type=bool, default=False,
        help='show the process while in training')
    parser.add_argument(
        '--algorithm', type=str, default='dqn',
        help='choose algorithm dqn, reinforce, a2c, ppo,super_dqn.')
    parser.add_argument(
        '--model', type=str, default='base',
        help='choose model base and FRAP.')
    parser.add_argument(
        '--gpu', type=bool, default=False,
        help='choose model base and FRAP.')
    parser.add_argument(
        '--replay_name', type=str, default=None,
        help='activate only in test mode and write file_name to load weights.')
    parser.add_argument(
        '--replay_epoch', type=str, default=None,
        help='activate only in test mode and write file_name to load weights.')
    return parser.parse_known_args(args)[0]


def train(flags, time_data, configs, sumoConfig):

    # check gui option
    if flags.disp == True:
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
        configs['action_size'] = 2
        configs['state_space'] = 8  # 4phase에서 각각 받아오는게 아니라 마지막에 한번만 받음
        configs['model'] = 'base'
    elif flags.model.lower() == 'base':
        configs['action_space'] = 13
        configs['action_size'] = 2
        configs['state_space'] = 8
        configs['model'] = 'base'
    elif flags.model.lower() == 'frap':
        configs['action_space'] = configs['num_phase']
        configs['action_size'] = 1
        configs['state_space'] = 16
        configs['model'] = 'frap'

    if flags.algorithm.lower() == 'dqn':
        from train import dqn_train
        from configs import DQN_TRAFFIC_CONFIGS
        configs = merge_dict_non_conflict(configs, DQN_TRAFFIC_CONFIGS)
        configs['time_size'] = int((torch.tensor(configs['phase_period'])
                                    - torch.tensor(configs['min_phase']).sum())/configs['num_phase'])  # 최대에서 최소 뺀 값이 size가 됨
        dqn_train(configs, time_data, sumoCmd)

    elif flags.algorithm.lower() == 'super_dqn':
        from train import super_dqn_train
        from configs import SUPER_DQN_TRAFFIC_CONFIGS
        configs = merge_dict_non_conflict(configs, SUPER_DQN_TRAFFIC_CONFIGS)
        super_dqn_train(configs, time_data, sumoCmd)


def test(flags, configs, sumoConfig):
    from Env.Env import TL3x3Env
    from Agent.dqn import Trainer
    from Env.MultiEnv import GridEnv
    from utils import save_params, load_params, update_tensorboard
    from test import dqn_test, super_dqn_test
    if flags.disp == True:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    sumoCmd = [sumoBinary, "-c", sumoConfig]

    if flags.algorithm.lower() == 'dqn':
        dqn_test(flags, sumoCmd, configs)
    elif flags.algorithm.lower() == 'super_dqn':
        super_dqn_test(flags, sumoCmd, configs)


def simulate(flags, configs, sumoConfig):
    if flags.disp == True:
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    sumoCmd = [sumoBinary, "-c", sumoConfig]
    MAX_STEPS = configs['max_steps']
    traci.start(sumoCmd)
    a = time.time()
    traci.simulation.subscribe([tc.VAR_ARRIVED_VEHICLES_NUMBER])
    # traci.edge.subscribe('n_2_2_to_n_2_1', [
    #                      tc.LAST_STEP_VEHICLE_HALTING_NUMBER], 0, 2000)
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
    b = time.time()
    traci.close()
    # edgesss = traci.edge.getSubscriptionResults('n_2_2_to_n_2_1')
    # print(edgesss)
    print('======== arrived number:{} avg waiting time:{},avg velocity:{}'.format(
        arrived_vehicles, avg_waiting_time/MAX_STEPS, avg_velocity))
    print("sim_time=", b-a)


def main(args):
    random_seed = 20000
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    flags = parse_args(args)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda and flags.gpu == True else "cpu")
    # device = torch.device('cpu')
    print("Using device: {}".format(device))
    configs = EXP_CONFIGS
    configs['device'] = str(device)
    configs['current_path'] = os.path.dirname(os.path.abspath(__file__))
    configs['mode'] = flags.mode.lower()
    time_data = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))
    configs['time_data'] = str(time_data)
    configs['file_name']=configs['time_data']

    # check the network
    configs['network']=flags.network.lower()
    if configs['network'] == 'grid':
        from Network.grid import GridNetwork  # network바꿀때 이걸로 바꾸세요(수정 예정)
        configs['grid_num'] = 3
        if configs['mode']=='simulate':
            configs['file_name'] = '{}x{}grid'.format(
            configs['grid_num'], configs['grid_num'])
        elif configs['mode']=='test': #test
            configs['file_name']=flags.network.lower()
        #Generating Network
        network = GridNetwork(configs, grid_num=configs['grid_num'])
        network.generate_cfg(True, configs['mode'])

        if flags.algorithm.lower() == 'super_dqn':
            # rl_list 설정
            side_list = ['u', 'r', 'd', 'l']
            tl_rl_list = list()
            for _, node in enumerate(configs['node_info']):
                if node['id'][-1] not in side_list:
                    tl_rl_list.append(node['id'])
            configs['tl_rl_list'] = tl_rl_list
    
    #Generating Network
    else:  # map file 에서 불러오기
        print("Load from map file")
        from Network.map import MapNetwork
        configs['grid_num'] = 3
        configs['num_lanes']=2
        configs['load_file_name']=configs['network']
        mapnet=MapNetwork(configs)
        MAP_CONFIGS=mapnet.get_tl_from_xml()
        for key in MAP_CONFIGS.keys():
            configs[key]=MAP_CONFIGS[key]

        mapnet.gen_net_from_xml()
        mapnet.gen_rou_from_xml()


    # check the environment
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # check the mode
    if configs['mode'] == 'train':
        # init train setting
        configs['num_agent'] = len(configs['tl_rl_list'])
        configs['max_phase_num'] = 4
        configs['offset'] = [0 for i in range(configs['num_agent'])]  # offset 임의 설정
        configs['tl_max_period'] = [160 for i in range(configs['num_agent'])] # max period 임의 설정
        configs['mode'] = 'train'
        sumoConfig = os.path.join(
            configs['current_path'], 'training_data', time_data, 'net_data', configs['file_name']+'_train.sumocfg')
        train(flags, time_data, configs, sumoConfig)
    elif configs['mode'] == 'test':
        configs['time_data'] = flags.replay_name
        configs['mode'] = 'test'
        sumoConfig = os.path.join(
            configs['current_path'], 'training_data', configs['file_name'],'net_data',configs['file_name']+'_test.sumocfg')
        test(flags, configs, sumoConfig)
    else:  # simulate
        configs['mode'] = 'simulate'
        sumoConfig = os.path.join(
            configs['current_path'], 'Net_data', configs['file_name']+'_simulate.sumocfg')
        simulate(flags, configs, sumoConfig)


if __name__ == '__main__':
    main(sys.argv[1:])

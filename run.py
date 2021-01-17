import argparse
import json
import os
import sys
import traci
import traci.constants as tc
import torch
from torch.utils.tensorboard import SummaryWriter
from gen_net import configs

from Env.env import TLEnv
from Agent.dqn import Trainer
from sumolib import checkBinary


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="choose the mode",
        epilog="python run.py mode")

    # required input parameters
    parser.add_argument(
        'mode', type=str,
        help='train or test')
    parser.add_argument(
        '--network', type=str, default='grid',
        help='choose network in Env')
    # optional input parameters
    parser.add_argument(
        '--disp', type=str, choices=['yes', 'no'], default='no',
        help='show the process while in training')
    return parser.parse_known_args(args)[0]


def train(flags, configs, sumoConfig):
    # check gui option
    if flags.disp == 'yes':
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    configs['mode'] = 'train'
    sumoBinary = checkBinary('sumo-gui')
    sumoCmd = [sumoBinary, "-c", sumoConfig, '--start']
    agent = Trainer(optimizer, configs)
    tl_rl_list = traci.trafficlight.getIDList()
    env = TLEnv(tl_rl_list, optimizer, configs)
    optimizer = optim.Adam(
        self.mainQNetwork.parameters(), lr=learning_rate)
    NUM_EPOCHS = configs['num_epochs']
    while epoch < NUM_EPOCHS:
        traci.start(sumoCmd)
        step = 0
        done = False
        # state initialization
        state = env.get_state()
        # agent setting
        writer = SummaryWriter(os.path.join(
            configs['current_path'], 'training_data'))
        total_reward = 0
        while step < MAX_STEP:
            '''
            # state=env.get_state(action) #partial하게는 env에서 조정
            action=agent.get_action(state)
            env.step(action)
            reward=env.get_reward()
            next_state=env.get_state()
            # if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
            step += 1
            state=next_state
            '''
            optimizer.zero_grad()  # env와 agent가 network를 공유하는 경우 step마다 최초 초기화
            action = agent.get_action(state)
            env.step(action)
            reward = env.get_reward()
            next_state = env.get_state()
            agent.save_replay(state, action, reward, next_state)
            '''
            추가사항

            store transition in D (experience replay)
            Sample random minibatch from D

            set yi

            '''
            state = next_state
            total_reward += reward
            step += 1
            if step == MAX_STEP:
                done = True
            agent.update(done)
            traci.simulationStep()

        loss = agent.get_loss()  # 총 loss
        writer.add_scalar('episode/loss', loss, step)  # 1 epoch마다
        writer.add_scalar('episode/reward', total_reward, step)  # 1 epoch마다
        print('{} epoch/ loss: {} return: {} '.format(epoch, loss, total_reward))
        traci.close()

    writer.close()


def test(flags, configs, sumoConfig):
    configs['mode'] = 'test'
    sumoBinary = checkBinary('sumo-gui')
    sumoCmd = [sumoBinary, "-c", sumoConfig, '--start']
    traci.start(sumoCmd)
    step = 0
    tls_id_list = traci.junction.getIDList()
    while step < 1000:
        traci.simulationStep()
        # if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
        step += 1
    traci.close()


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device: {}".format(device))
    configs['current_path'] = os.path.dirname(os.path.abspath(__file__))
    configs['device'] = device
    flags = parse_args(args)
    configs['mode'] = flags.mode.lower()

    # check the network
    if flags.network.lower() == 'grid':
        from grid import GridNetwork  # network바꿀때 이걸로 바꾸세요(수정 예정)
        configs['grid_num'] = 4
        print(configs['grid_num'])
        network = GridNetwork(configs, grid_num=configs['grid_num'])
        print(configs['grid_num'])
        network.generate_cfg(True, configs['mode'])
    # check the mode
    if configs['mode'] == 'train':
        configs['mode'] = 'train'
        sumoConfig = os.path.join(
            configs['current_path'], 'Env', configs['file_name']+'_train.sumocfg')
        train(flags, configs, sumoConfig)
    elif configs['mode'] == 'test':
        configs['mode'] = 'test'
        sumoConfig = os.path.join(
            configs['current_path'], 'Env', configs['file_name']+'_test.sumocfg')
        test(flags, configs, sumoConfig)

    # check the environment
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")


if __name__ == '__main__':
    main(sys.argv[1:])

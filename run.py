import argparse
import json
import os
import sys
import traci
import traci.constants as tc
import torch
from torch.utils.tensorboard import SummaryWriter
from gen_net import configs
import torch.optim as optim
from Env.env import TLEnv
from Agent.dqn import Trainer
from sumolib import checkBinary
import time

def mappingMovement(movement):
    if movement == 'G':
        return 1
    elif movement == 'r':
        return 0
    else:
        return -1  # error


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
        '--disp', type=str, default='no',
        help='show the process while in training')
    return parser.parse_known_args(args)[0]


def train(flags, configs, sumoConfig):
    # check gui option
    if flags.disp == 'yes':
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    configs['mode'] = 'train'
    sumoCmd = [sumoBinary, "-c", sumoConfig, '--start']
    agent = Trainer(configs)
    tl_rl_list = ['n_1_1']
    NUM_EPOCHS = configs['num_epochs']
    epoch = 0
    MAX_STEPS = configs['max_steps']
    writer = SummaryWriter(os.path.join(
        configs['current_path'], 'training_data',time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))))

    while epoch < NUM_EPOCHS:
        traci.start(sumoCmd)
        env = TLEnv(tl_rl_list,configs)
        step = 0
        loss=0
        done = False
        # state initialization
        state = env.get_state()
        # agent setting
        total_reward = 0
        while step < MAX_STEPS:
            '''
            # state=env.get_state(action) #partial하게는 env에서 조정
            action=agent.get_action(state)
            env.step(action)
            reward=env.get_reward()
            next_state=env.get_state()
            # if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
            step += 1
            state=next_state
            
            store transition in D (experience replay)
            Sample random minibatch from D

            set yi
            '''

            action = agent.get_action(state)
            env.step(action)  # action 적용함수
            for _ in range(20): # 10초마다 행동 갱신
                env.collect_state()
                traci.simulationStep()
                step += 1

            traci.trafficlight.setRedYellowGreenState(tl_rl_list[0],'y'*28)
            for _ in range(5):
                traci.simulationStep()
                env.collect_state()
                step += 1

            reward = env.get_reward()
            next_state = env.get_state()
            agent.save_replay(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            step += 1
            if step == MAX_STEPS:
                done = True
            agent.update(done)
            loss += agent.get_loss()  # 총 loss
            traci.simulationStep()
            if step%200==0:
                agent.target_update()

        epoch+=1
        writer.add_scalar('episode/loss', loss, step*epoch)  # 1 epoch마다
        writer.add_scalar('episode/reward', total_reward, step*epoch)  # 1 epoch마다
        writer.flush()
        print('======== {} epoch/ loss: {} return: {} '.format(epoch,loss, total_reward))
        traci.close()

    writer.close()


def test(flags, configs, sumoConfig):
    configs['mode'] = 'test'
    sumoBinary = checkBinary('sumo-gui')
    sumoCmd = [sumoBinary, "-c", sumoConfig]
    traci.start(sumoCmd)
    step = 0
    tls_id_list = traci.trafficlight.getIDList()
    edge_list = traci.edge.getIDList()
    while step < 1000:
        phase = list()
        traci.simulationStep()
        inflow = 0
        outflow = 0
        for _, edge in enumerate(edge_list):  # 이 부분을 밖에서 list로 구성해오면 쉬움
            if edge[-5:] == 'n_2_2':  # outflow 여기에 n_2_2대신에 tl_id를 넣으면 pressure가 되는 것
                inflow += traci.edge.getLastStepVehicleNumber(edge)
            elif edge[:5] == 'n_2_2':  # inflow
                outflow += traci.edge.getLastStepVehicleNumber(edge)
        phase = traci.trafficlight.getRedYellowGreenState('n_2_2')
        print(phase)
        state = torch.zeros(8, dtype=torch.int16)
        for i in range(4):  # 4차로
            phase = phase[1:]  # 우회전
            state[i] = mappingMovement(phase[0])  # 직진신호 추출
            phase = phase[3:]  # 직전
            state[i+1] = mappingMovement(phase[0])  # 좌회전신호 추출
            phase = phase[1:]  # 좌회전
            phase = phase[1:]  # 유턴
        print(state)
        print('in: {} out: {}, pressure: {}'.format(
            inflow, outflow, inflow-outflow))
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
        configs['grid_num'] = 3
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

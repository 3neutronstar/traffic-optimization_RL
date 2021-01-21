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
from Env.env import TL1x1Env
from Agent.dqn import Trainer
from sumolib import checkBinary
import time
from Env.GridEnv import GridEnv
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


def save_params(configs, time_data):
    with open(os.path.join(configs['current_path'], 'training_data', '{}_{}.json'.format(configs['file_name'], time_data)), 'w') as fp:
        json.dump(configs, fp, indent=2)


def load_params(configs, file_name):
    ''' replay_name from flags.replay_name '''
    with open(os.path.join(configs['current_path'], 'training_data', '{}.json'.format(file_name)), 'r') as fp:
        configs = json.load(fp)
    return configs


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="choose the mode",
        epilog="python run.py mode")

    # required input parameters
    parser.add_argument(
        'mode', type=str,
        help='train or test, simulate')
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
    return parser.parse_known_args(args)[0]


def train(flags, configs, sumoConfig):
    # init train setting
    time_data = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))
    # check gui option
    if flags.disp == 'yes':
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')
    sumoCmd = [sumoBinary, "-c", sumoConfig, '--start']
    # configs setting
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
        env = TL1x1Env(tl_rl_list, configs)
        # env = GridEnv( configs)
        step = 0
        loss = 0
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
            loss += agent.get_loss()  # 총 loss
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
            if step % 1000 == 0:
                agent.target_update()  # dqn
                agent.update_hyperparams() # lr and epsilon upate

        traci.close()
        epoch += 1
        writer.add_scalar('episode/loss', loss/MAX_STEPS,
                          step*epoch)  # 1 epoch마다
        writer.add_scalar('episode/reward', total_reward,
                          step*epoch)  # 1 epoch마다
        writer.add_scalar('episode/arrived_num', arrived_vehicles,
                          step*epoch)  # 1 epoch마다
        writer.flush()
        print('======== {} epoch/ loss: {} return: {} arrived number:{}'.format(epoch,
                                                                                loss/MAX_STEPS, total_reward, arrived_vehicles))
        if epoch % 50 == 0:
            agent.save_weights(
                configs['file_name']+'_{}_{}'.format(time_data, epoch))

    writer.close()


def test(flags, configs, sumoConfig):
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
    env = TL1x1Env(tl_rl_list, configs)
    step = 0
    # state initialization
    state = env.get_state()
    # agent setting
    total_reward = 0
    arrived_vehicles = 0
    with torch.no_grad():
        while step < MAX_STEPS:

            action = agent.get_action(state, reward)
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
            if step % 200 == 0:
                agent.target_update()

        traci.close()
        print('======== return: {} arrived number:{}'.format(
            total_reward, arrived_vehicles))


def simulate(flags, configs, sumoConfig):
    sumoBinary = checkBinary('sumo')
    sumoCmd = [sumoBinary, "-c", sumoConfig]
    MAX_STEPS = configs['max_steps']
    tl_rl_list = configs['tl_rl_list']
    traci.start(sumoCmd)
    traci.simulation.subscribe([tc.VAR_ARRIVED_VEHICLES_NUMBER])
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
        vehicle_list = traci.vehicle.getIDList()
        for i, vehicle in enumerate(vehicle_list):
            speed = traci.vehicle.getSpeed(vehicle)
            avg_velocity = float((i)*avg_velocity+speed) / \
                float(i+1)  # incremental avg

        arrived_vehicles += traci.simulation.getAllSubscriptionResults()[
            ''][0x79]  # throughput

    traci.close()
    print('======== arrived number:{} avg waiting time:{},avg velocity:{}'.format(
        arrived_vehicles, avg_waiting_time/MAX_STEPS, avg_velocity))


def main(args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device: {}".format(device))
    configs['current_path'] = os.path.dirname(os.path.abspath(__file__))
    configs['device'] = str(device)
    flags = parse_args(args)
    configs['mode'] = flags.mode.lower()

    # check the network
    if flags.network.lower() == 'grid':
        from grid import GridNetwork  # network바꿀때 이걸로 바꾸세요(수정 예정)
        configs['grid_num'] = 3
        configs['file_name'] = '{}x{}grid'.format(
            configs['grid_num'], configs['grid_num'])
        configs['grid_side'] = 'in'  # out mode도 만들 예정 in모드시에 내부 tl만 컨트롤
        network = GridNetwork(configs, grid_num=configs['grid_num'])
        network.generate_cfg(True, configs['mode'])
    # check the mode
    if configs['mode'] == 'train':
        configs['mode'] = 'train'
        sumoConfig = os.path.join(
            configs['current_path'], 'Net_data', configs['file_name']+'_train.sumocfg')
        train(flags, configs, sumoConfig)
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

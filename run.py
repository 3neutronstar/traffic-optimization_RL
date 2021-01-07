import argparse
import json, os, sys
import traci
import traci.constants as tc
from grid_make3x3 import configs
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

    # optional input parameters
    parser.add_argument(
        '--disp',type=str, choices=['yes','no'],default='no',
        help='show the process while in training')
    return parser.parse_known_args(args)[0]

def train(flags):
    sumoBinary = checkBinary('sumo-gui')
    sumoCmd = [sumoBinary, "-c", "{}.sumocfg".format(configs['file_name'])]
    traci.start(sumoCmd)
    step = 0
    while step < 1000:
        traci.simulationStep()
        # if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
        step += 1
    traci.close()  


def test(flags):
    sumoBinary = checkBinary('sumo-gui')
    sumoCmd = [sumoBinary, "-c", "{}.sumocfg".format(configs['file_name'])]
    traci.start(sumoCmd)
    step = 0
    tls_id_list=traci.junction.getIDList()
    print(tls_id_list)
    while step < 1000:
        traci.simulationStep()
        # if traci.inductionloop.getLastStepVehicleNumber("0") > 0:
        step += 1
    traci.close()  

def main(args):
    flags=parse_args(args)
    if flags.disp == 'yes':
        sumoBinary = checkBinary('sumo-gui')
    else:
        sumoBinary = checkBinary('sumo')

    if flags.mode.lower()=='train':
        train(flags)
    elif flags.mode.lower()=='test':
        test(flags)
    
    #check the environment
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")
    


if __name__ =='__main__':
    main(sys.argv[1:])
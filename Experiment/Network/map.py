from gen_net import Network
from configs import EXP_CONFIGS
import math
import argparse
import json
import os
import sys
import time
from xml.etree.ElementTree import parse


class MapNetwork(Network):
    def __init__(self, configs):
        super().__init__(configs)
        self.tl_rl_list = list()
        self.offset_list = list()
        self.phase_list = list()
        self.common_phase = list()

    def specify_traffic_light(self):
        self.traffic_light = traffic_light

        return traffic_light

    def get_tl_from_xml(self):
        # , 'Network') # 가동시
        file_path = os.path.join(self.configs['current_path'])
        tl_tree = parse(file_path)
        tlLogicList = tl_tree.findall('tlLogic')
        for tlLogic in tlLogicList:
            self.offset_list.append(tlLogic.attrib['offset'])
            self.tl_rl_list.append(tlLogic.attrib['id'])  # rl 조종할 tl_rl추가
            phaseList = tlLogic.findall('phase')
            phase_state_list = list()
            phase_duration_list = list()
            phase_period = 0
            for phase in phaseList:
                phase_state_list.append(phase.attrib['state'])
                phase_duration_list.append(int(phase.attrib['duration']))
                phase_period += int(phase.attrib['duration'])
            self.phase_list.append(phase_state_list)
            self.common_phase.append(phase_duration_list)
        configs={
            'common_phase':self.common_phase,
            'tl_rl_list':self.tl_rl_list,
            'offset':self.offset_list,
            'phase_list':self.phase_list,
        }
        return configs

    def print(self):
        print('all')


if __name__ == "__main__":
    configs['current_path'] = os.path.abspath(__file__)
    mapnet=MapNetwork(configs)
    mapnet.generate_cfg(False)
    mapnet.sumo_gui()

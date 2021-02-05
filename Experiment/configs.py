import torch
Edges = list()
Nodes = list()
Vehicles = list()
EXP_CONFIGS = {
    'num_lanes': 3,
    'model': 'normal',
    'file_name': '4x4grid',
    'tl_rl_list': ['n_1_1'],
    'laneLength': 300.0,
    'num_cars': 1800,
    'flow_start': 0,
    'flow_end': 3600,
    'sim_start': 0,
    'max_steps': 3600,
    'num_epochs': 1000,
    'edge_info': Edges,
    'node_info': Nodes,
    'vehicle_info': Vehicles,
    'mode': 'simulate',
}
# DQN
DQN_TRAFFIC_CONFIGS = {
    # 1,agent,num_phase순서
    'min_phase': [[[20, 20, 20, 20]]], #각 phase 최소 길이
    # 1,agent,num_phase순서
    'max_phase': [[[50, 50, 50, 50]]], # 각 phase최대 길이
    # 1, agent순서
    'tl_period': [[160]], # 각 tl이 가지는 period

}
# Decentralized_DQN
SUPER_DQN_TRAFFIC_CONFIGS = {
    # 1,agent,num_phase순서
    'min_phase': [[20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20]],
    'common_phase': [[40, 40, 40, 40], [40, 40, 40, 40], [40, 40, 40, 40], [40, 40, 40, 40], [40, 40, 40, 40], [40, 40, 40, 40], [40, 40, 40, 40], [40, 40, 40, 40], [40, 40, 40, 40]],
    # 1,agent,num_phase순서
    'max_phase': [[49, 49, 49, 49], [49, 49, 49, 49], [49, 49, 49, 49], [49, 49, 49, 49], [49, 49, 49, 49], [49, 49, 49, 49], [49, 49, 49, 49], [49, 49, 49, 49], [49, 49, 49, 49]],
    # 1, agent순서
    'tl_period': [[160],[160],[160],[160],[160],[160],[160],[160],[160]],
    'matrix_actions': [[0, 0, 0, 0], [1, 0, 0, -1], [1, 0, -1, 0], [1, -1, 0, 0], [0, 1, 0, -1], [0, 1, -1, 0], [0, 0, 1, -1],
                       [1, 0, 0, -1], [1, 0, -1, 0], [1, 0, 0, -1], [0, 1, 0, -1], [0, 1, -1, 0], [0, 0, 1, -1]]

}

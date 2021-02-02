import torch
Edges = list()
Nodes = list()
Vehicles = list()
EXP_CONFIGS = {
    'num_lanes': 2,
    'model': 'normal',
    'file_name': '4x4grid',
    'tl_rl_list': ['n_1_1'],
    'laneLength': 300.0,
    'num_cars': 1800,
    'flow_start': 0,
    'flow_end': 3200,
    'sim_start': 0,
    'max_steps': 3200,
    'num_epochs': 1000,
    'edge_info': Edges,
    'node_info': Nodes,
    'vehicle_info': Vehicles,
    'mode': 'simulate',
}

TRAFFIC_CONFIGS = {
    # 1,agent,num_phase순서
    'min_phase': [[[20,20,20,20]]],
    # 1,agent,num_phase순서
    'max_phase': [[[50,50,50,50]]],
    # 1, agent순서
    'phase_period': [[160]],

}

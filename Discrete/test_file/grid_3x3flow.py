from gen_net import Network
from gen_net import configs

node_id = ["n_nl", "n_nc", "n_nr",
           "n_ln", "n_cnl", "n_cnc", "n_cnr", "n_rn",
           "n_lc", "n_ccl", "n_ccc", "n_ccr", "n_rc",
           "n_ls", "n_csl", "n_csc", "n_csr", "n_rs",
           "n_sl", "n_sc", "n_sr"]

edge_dict = {
    "n_nl": ["n_cnl"],
    "n_nc": ["n_cnc"],
    "n_nr": ["n_cnr"],
    "n_ln": ["n_cnl"],
    "n_lc": ["n_ccl"],
    "n_ls": ["n_csl"],
    "n_rn": ["n_cnr"],
    "n_rc": ["n_ccr"],
    "n_rs": ["n_csr"],
    "n_sl": ["n_csl"],
    "n_sc": ["n_csc"],
    "n_sr": ["n_csr"],
    "n_cnl": ["n_nl", "n_ln", "n_cnr", "n_csl"],
    "n_cnc": ["n_nc", "n_ccc", "n_cnl", "n_cnr"],
    "n_cnr": ["n_nr", "n_cnl", "n_rn", "n_csr"],
    "n_csl": ["n_ccl", "n_ls", "n_csc", "n_sl"],
    "n_csc": ["n_csl", "n_ccc", "n_sc", "n_csr"],
    "n_csr": ["n_csc", "n_ccr", "n_rs", "n_sr"],
    "n_ccl": ["n_lc", "n_csl", "n_cnl", "n_ccc"],
    "n_ccc": ["n_ccl", "n_cnc", "n_ccr", "n_csc"],
    "n_ccr": ["n_ccc", "n_cnr", "n_rc", "n_csr"],
}


class Grid3x3network(Network):
    def __init__(self, node_id, edge_dict, configs):
        super().__init__(configs)
        self.node_id = node_id
        self.edge_dict = edge_dict

    def specify_edge(self):
        edges = list()
        for _, dict_key in enumerate(self.edge_dict.keys()):
            for i, _ in enumerate(self.edge_dict[dict_key]):
                edges_dict = dict()
                edges_dict = {
                    'from': dict_key,
                    'id': "{}_to_{}".format(dict_key, self.edge_dict[dict_key][i]),
                    'to': self.edge_dict[dict_key][i],
                    'numLanes': self.num_lanes
                }
                edges.append(edges_dict)
        self.edges = edges
        return edges

    def specify_node(self):
        node_list = list()
        node_dict = dict()
        for _, node in enumerate(self.node_id):
            node_dict['id'] = node
            # y value decision
            if node[2] == 'n':
                y = 2.0*self.laneLength
            elif node[2] == 's':
                y = -2.0*self.laneLength
            else:  # node[2]=='c'
                node_dict['type'] = 'traffic_light'  # n_c 계열은 모두 신호등
                node_dict['tl'] = node
                if node[3] == 'n':
                    y = self.laneLength
                elif node[3] == 's':
                    y = (-1.0)*self.laneLength
                else:  # node[3]=='c'
                    y = 0.0
            # x value decision
            if node[2] == 'l':
                x = -2.0*self.laneLength
            elif node[2] == 'r':
                x = 2.0*self.laneLength
            else:  # node[2]='c'
                try:
                    if node[-1] == 'l':
                        x = (-1.0)*self.laneLength
                    elif node[-1] == 'r':
                        x = self.laneLength
                    else:  # node[-1]=='c'
                        x = 0.0
                except IndexError as e:
                    raise NotImplementedError from e
            node_dict['x'] = str('%.1f' % x)
            node_dict['y'] = str('%.1f' % y)
            node_list.append(node_dict)
            node_dict = dict()
        self.node = node_list
        return node_list

    def specify_flow(self):
        edge_start = list()
        node_fin = list()
        flow_dict = dict()
        edge_fin = list()
        for _, key_start in enumerate(self.edge_dict):
            if len(self.edge_dict[key_start]) == 1:
                append_start_str = "{}_to_{}".format(
                    key_start, self.edge_dict[key_start][0])
                edge_start.append(append_start_str)
                if key_start[2] == 'n':
                    key_fin = key_start[:2]+'s'+key_start[3:]
                    for _, dict_key in enumerate(self.edge_dict.keys()):
                        for i in range(len(self.edge_dict[dict_key])):
                            if self.edge_dict[dict_key][i] == key_fin:
                                node_fin = dict_key
                    append_fin_str = "{}_to_{}".format(node_fin, key_fin)
                elif key_start[2] == 's':
                    key_fin = key_start[:2]+'n'+key_start[3:]
                    for _, dict_key in enumerate(self.edge_dict.keys()):
                        for i in range(len(self.edge_dict[dict_key])):
                            if self.edge_dict[dict_key][i] == key_fin:
                                node_fin = dict_key
                    append_fin_str = "{}_to_{}".format(node_fin, key_fin)
                elif key_start[2] == 'r':
                    key_fin = key_start[:2]+'l'+key_start[3:]
                    for _, dict_key in enumerate(self.edge_dict.keys()):
                        for i in range(len(self.edge_dict[dict_key])):
                            if self.edge_dict[dict_key][i] == key_fin:
                                node_fin = dict_key
                    append_fin_str = "{}_to_{}".format(node_fin, key_fin)
                elif key_start[2] == 'l':
                    key_fin = key_start[:2]+'r'+key_start[3:]
                    for _, dict_key in enumerate(self.edge_dict.keys()):
                        for i in range(len(self.edge_dict[dict_key])):
                            if self.edge_dict[dict_key][i] == key_fin:
                                node_fin = dict_key
                    append_fin_str = "{}_to_{}".format(node_fin, key_fin)
                edge_fin.append(append_fin_str)
                flow_dict = {'id': key_start, 'from': append_start_str, 'to': append_fin_str,
                             'begin': self.flow_start, 'end': self.flow_end, 'number': self.num_cars}
                self.flows.append(flow_dict)
                flow_dict = dict()

        return self.flows


a = Grid3x3network(node_id, edge_dict, configs)
a.sumo_gui()

import xml.etree.cElementTree as ET
import os
laneLength = 100.0
num_lanes = 3
file_name = '3x3intersection'

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


class Network():
    def __init__(self, node_id, edge_dict, file_name, numLanes):
        self.node_id = node_id
        self.edge_dict = edge_dict
        self.xml_node_name = file_name+'.nod'
        self.xml_edg_name = file_name+'.edg'
        self.current_path = os.getcwd()
        self.node = self.specify_node()  # node info
        self.num_lanes = numLanes

    def specify_edge(self):
        edges = list()
        for _, dict_key in enumerate(self.edge_dict.keys()):
            for i, _ in enumerate(self.edge_dict[dict_key]):
                print('    <edge from="{}" id="{}_to_{}" to="{}" numLanes="{}"/>'.format(dict_key,
                                                                                         dict_key,
                                                                                         self.edge_dict[dict_key][i],
                                                                                         self.edge_dict[dict_key][i],
                                                                                         num_lanes))
                edges.append("{}_to_{}".format(
                    dict_key, self.edge_dict[dict_key][i]))
        return edges

    def specify_node(self):
        node_list = list()
        node_dict = dict()
        for node in self.node_id:
            node_dict['node id'] = node
            # y value decision
            if node[2] == 'n':
                y = 2.0*laneLength
            elif node[2] == 's':
                y = -2.0*laneLength
            else:  # node[2]=='c'
                node_dict['type'] = 'traffic_light'  # n_c 계열은 모두 신호등
                node_dict['tl'] = node
                if node[3] == 'n':
                    y = laneLength
                elif node[3] == 's':
                    y = (-1.0)*laneLength
                else:  # node[3]=='c'
                    y = 0
            # x value decision
            if node[2] == 'l':
                x = -2.0*laneLength
            elif node[2] == 'r':
                x = 2.0*laneLength
            else:  # node[2]='c'
                try:
                    if node[-1] == 'l':
                        x = (-1.0)*laneLength
                    elif node[-1] == 'r':
                        x = laneLength
                    else:  # node[-1]=='c'
                        x = 0
                except IndexError as e:
                    raise NotImplementedError from e
            node_dict['x'] = x
            node_dict['y'] = y
            node_list.append(node_dict)
        self.node = node_list
        return node_list

    def generate_nod_xml(self):
        xml_pos = os.path.join(self.current_path, self.xml_node_name)
        nod_xml = ET.Element('nodes')
        for node_dict in self.node:
            if node_dict['type'] is not None:
                doc = ET.SubElement(
                    nod_xml, 'node', id=node_dict['id'], x=node_dict['x'], y=node_dict['y'], type=node_dict['type'], tl=node_dict['id'])
            else:
                doc = ET.SubElement(
                    nod_xml, 'node', id=node_dict['id'], x=node_dict['x'], y=node_dict['y'])

    def generate_edg_xml(self):
        xml_pos = os.path.join(self.current_path, self.xml_edg_name)
        edg_xml = ET.Element('edges')
        for _, dict_key in enumerate(self.edge_dict.keys()):
            for i, _ in enumerate(self.edge_dict[dict_key]):
                print_str = 'edge from="{}" id="{}_to_{}" to="{}" numLanes="{}"/>'.format(dict_key,
                                                                                          dict_key, edge_dict[dict_key][i], edge_dict[dict_key][i], self.num_lanes)
                doc = ET.SubElement(edg_xml, print_str)
                print('edge from="{}" id="{}_to_{}" to="{}" numLanes="{}"/>'.format(dict_key,
                                                                                    dict_key, edge_dict[dict_key][i], edge_dict[dict_key][i], self.num_lanes))


def specify_flow(edge_dict):
    edge_start = list()
    node_fin = list()
    flows = list()
    edge_fin = list()
    for _, key_start in enumerate(edge_dict):
        if len(edge_dict[key_start]) == 1:
            append_start_str = "{}_to_{}".format(
                key_start, edge_dict[key_start][0])
            edge_start.append(append_start_str)
            if key_start[2] == 'n':
                key_fin = key_start[:2]+'s'+key_start[3:]
                for _, dict_key in enumerate(edge_dict.keys()):
                    for i in range(len(edge_dict[dict_key])):
                        if edge_dict[dict_key][i] == key_fin:
                            node_fin = dict_key
                append_fin_str = "{}_to_{}".format(node_fin, key_fin)
            elif key_start[2] == 's':
                key_fin = key_start[:2]+'n'+key_start[3:]
                for _, dict_key in enumerate(edge_dict.keys()):
                    for i in range(len(edge_dict[dict_key])):
                        if edge_dict[dict_key][i] == key_fin:
                            node_fin = dict_key
                append_fin_str = "{}_to_{}".format(node_fin, key_fin)
            elif key_start[2] == 'r':
                key_fin = key_start[:2]+'l'+key_start[3:]
                for _, dict_key in enumerate(edge_dict.keys()):
                    for i in range(len(edge_dict[dict_key])):
                        if edge_dict[dict_key][i] == key_fin:
                            node_fin = dict_key
                append_fin_str = "{}_to_{}".format(node_fin, key_fin)
            elif key_start[2] == 'l':
                key_fin = key_start[:2]+'r'+key_start[3:]
                for _, dict_key in enumerate(edge_dict.keys()):
                    for i in range(len(edge_dict[dict_key])):
                        if edge_dict[dict_key][i] == key_fin:
                            node_fin = dict_key
                append_fin_str = "{}_to_{}".format(node_fin, key_fin)
            edge_fin.append(append_fin_str)
            flows.append('from="{}" to="{}"'.format(
                append_start_str, append_fin_str))
            print('    <flow id="{}" from="{}" to="{}" begin="0" end="9000" number="1800" />'.format(
                key_start, append_start_str, append_fin_str))
    return flows


if __name__ == '__main__':
    network = Network(node_id, edge_dict, file_name, num_lanes)
    node_list = specify_node(node_id)
    flow_list = specify_flow(edge_dict)
    edge_list = specify_edgeid(edge_dict)

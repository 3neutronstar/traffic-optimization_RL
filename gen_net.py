import xml.etree.cElementTree as ET
from xml.etree.ElementTree import dump
from lxml import etree as ET
import os
E = ET.Element
Edges = list()
Nodes = list()
Vehicles = list()
configs = {
    'num_lanes': 3,
    'file_name': '4x4grid',
    'laneLength': 300.0,
    'num_cars': 1800,
    'flow_start': 0,
    'flow_end': 900,
    'sim_start': 0,
    'max_steps': 1000,
    'num_epochs': 3000,
    'edge_info': Edges,
    'node_info': Nodes,
    'vehicle_info': Vehicles,
    'mode': 'simulate',
    'learning_rate': 5e-5,
    'num_epochs': 3000,
    'gamma': 0.99,
    'tau':0.995,
    'model': 'normal',
    'batch_size': 32,
    'experience_replay_size': 1e5,
    'input_size': 16,
    'output_size': 8,
    'action_space': 8,
}


def indent(elem, level=0):
    i = "\n  " + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + ""
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i


class Network():
    def __init__(self, configs):
        self.configs = configs
        self.sim_start = self.configs['sim_start']
        self.max_steps = self.configs['max_steps']
        self.file_name = self.configs['file_name']
        self.xml_node_name = self.configs['file_name']+'.nod'
        self.xml_edg_name = self.configs['file_name']+'.edg'
        self.xml_con_name = self.configs['file_name']+'.con'
        self.xml_route_pos = self.file_name+'.rou'
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.current_Env_path = os.path.join(self.current_path, 'Env')
        self.num_cars = str(self.configs['num_cars'])
        self.num_lanes = str(self.configs['num_lanes'])
        self.flow_start = str(self.configs['flow_start'])
        self.flow_end = str(self.configs['flow_end'])
        self.laneLength = self.configs['laneLength']
        self.nodes = list()
        self.flows = list()
        self.vehicles = list()
        self.edges = list()
        self.connections = list()
        self.outputData = list()
        self.traffic_light = list()
        if self.configs['mode'] == 'test':
            self.generate_cfg(True, 'test')
        if self.configs['mode'] == 'train':
            self.generate_cfg(True, 'train')

    def specify_edge(self):
        edges = list()
        '''
        상속을 위한 함수
        '''
        return edges

    def specify_node(self):
        nodes = list()
        '''
        상속을 위한 함수
        '''

        return nodes

    def specify_flow(self):
        flows = list()
        '''
        상속을 위한 함수
        '''

        return flows

    def specify_connection(self):
        connections = list()
        '''
        상속을 위한 함수
        '''
        return connections

    def specify_outdata(self):
        outputData = list()
        '''
        상속을 위한 함수
        '''
        return outputData

    def specify_traffic_light(self):
        traffic_light = list()
        '''
        상속을 위한 함수
        '''
        return traffic_light

    def _generate_nod_xml(self):
        self.nodes = self.specify_node()
        nod_xml = ET.Element('nodes')

        for node_dict in self.nodes:
            # node_dict['x']=format(node_dict['x'],'.1f')
            nod_xml.append(E('node', attrib=node_dict))
            indent(nod_xml, 1)
        dump(nod_xml)
        tree = ET.ElementTree(nod_xml)
        # tree.write(self.xml_node_name+'.xml',encoding='utf-8',xml_declaration=True)
        tree.write(os.path.join(self.current_Env_path, self.xml_node_name+'.xml'), pretty_print=True,
                   encoding='UTF-8', xml_declaration=True)

    def _generate_edg_xml(self):
        self.edges = self.specify_edge()
        edg_xml = ET.Element('edges')
        for _, edge_dict in enumerate(self.edges):
            edg_xml.append(E('edge', attrib=edge_dict))
            indent(edg_xml, 1)
        dump(edg_xml)
        tree = ET.ElementTree(edg_xml)
        # tree.write(self.xml_edg_name+'.xml',encoding='utf-8',xml_declaration=True)
        tree.write(os.path.join(self.current_Env_path, self.xml_edg_name+'.xml'), pretty_print=True,
                   encoding='UTF-8', xml_declaration=True)

    def _generate_net_xml(self):
        # file_name_str=os.path.join(self.current_Env_path,self.file_name)
        file_name_str = os.path.join(self.current_Env_path, self.file_name)
        if len(self.connections) == 0:
            os.system('netconvert -n {}.nod.xml -e {}.edg.xml -o {}.net.xml'.format(
                file_name_str, file_name_str, file_name_str))
        else:  # connection이 존재하는 경우 -x
            os.system('netconvert -n {}.nod.xml -e {}.edg.xml -x {}.con.xml -o {}.net.xml'.format(
                file_name_str, file_name_str, file_name_str, file_name_str))

    def _generate_rou_xml(self):
        self.flows = self.specify_flow()
        route_xml = ET.Element('routes')
        if len(self.vehicles) != 0:  # empty
            for _, vehicle_dict in enumerate(self.vehicles):
                route_xml.append(E('veh', attrib=vehicle_dict))
                indent(route_xml, 1)
        if len(self.flows) != 0:
            for _, flow_dict in enumerate(self.flows):
                route_xml.append(E('flow', attrib=flow_dict))
                indent(route_xml, 1)
        dump(route_xml)
        tree = ET.ElementTree(route_xml)
        tree.write(os.path.join(self.current_Env_path, self.xml_route_pos+'.xml'), pretty_print=True,
                   encoding='UTF-8', xml_declaration=True)

    def _generate_con_xml(self):
        self.cons = self.specify_connection()
        self.xml_con_pos = self.xml_con_name
        con_xml = ET.Element('connections')
        if len(self.connections) != 0:  # empty
            for _, connection_dict in enumerate(self.connections):
                con_xml.append(E('connection', attrib=connection_dict))
                indent(con_xml, 1)

        dump(con_xml)
        tree = ET.ElementTree(con_xml)
        tree.write(os.path.join(self.current_Env_path, self.xml_con_pos+'.xml'), pretty_print=True,
                   encoding='UTF-8', xml_declaration=True)

    def generate_cfg(self, route_exist, mode='simulate'):
        self._generate_nod_xml()
        self._generate_edg_xml()
        self._generate_net_xml()
        self._generate_add_xml()
        sumocfg = ET.Element('configuration')
        inputXML = ET.SubElement(sumocfg, 'input')
        inputXML.append(
            E('net-file', attrib={'value': self.file_name+'.net.xml'}))
        indent(sumocfg)
        if route_exist == True:
            self._generate_rou_xml()
            if os.path.exists(os.path.join(self.current_Env_path, self.file_name+'.rou.xml')):
                inputXML.append(
                    E('route-files', attrib={'value': self.file_name+'.rou.xml'}))
                indent(sumocfg)

        if os.path.exists(os.path.join(self.current_Env_path, self.file_name+'.add.xml')):
            inputXML.append(
                E('additional-files', attrib={'value': self.file_name+'.add.xml'}))
            indent(sumocfg)

        time = ET.SubElement(sumocfg, 'time')
        time.append(E('begin', attrib={'value': str(self.sim_start)}))
        indent(sumocfg)
        time.append(E('end', attrib={'value': str(self.max_steps)}))
        indent(sumocfg)
        outputXML = ET.SubElement(sumocfg, 'output')
        # outputXML.append(
        #     E('netstate-dump', attrib={'value': os.path.join(self.current_Env_path, self.file_name+'_dump.net.xml')}))
        indent(sumocfg)
        dump(sumocfg)
        tree = ET.ElementTree(sumocfg)
        if mode == 'simulate':
            tree.write(os.path.join(self.current_Env_path, self.file_name+'_simulate'+'.sumocfg'),
                       pretty_print=True, encoding='UTF-8', xml_declaration=True)
        elif mode == 'test':
            tree.write(os.path.join(self.current_Env_path, self.file_name+'_test'+'.sumocfg'),
                       pretty_print=True, encoding='UTF-8', xml_declaration=True)
        elif mode == 'train':
            tree.write(os.path.join(self.current_Env_path, self.file_name+'_train'+'.sumocfg'),
                       pretty_print=True, encoding='UTF-8', xml_declaration=True)

    def _generate_add_xml(self):
        self.traffic_light = self.specify_traffic_light()

        additional = ET.Element('additional')
        # edgeData와 landData파일의 생성위치는 data
        additional.append(E('edgeData', attrib={'id': 'edgeData_00', 'file': '{}_edge.xml'.format(self.current_path+'/data/'+self.file_name), 'begin': '0', 'end': str(
            self.configs['max_steps']), 'freq': '1000'}))
        indent(additional, 1)
        additional.append(E('laneData', attrib={'id': 'laneData_00', 'file': '{}_lane.xml'.format(self.current_path+'/data/'+self.file_name), 'begin': '0', 'end': str(
            self.configs['max_steps']), 'freq': '1000'}))
        indent(additional, 1)
        if len(self.traffic_light) != 0:
            tlLogic = ET.SubElement(additional, 'tlLogic', attrib={
                                    'programID': '{}'.format('myprogram'), 'offset': '0', 'type': 'static'})
            tlLogic.append(E('phase', attrib={'duration': '{}'.format(
                '42'), 'state': 'rrrrrrrrrrrrrrrrrrr'}))
            tlLogic.append(E('phase', attrib={'duration': '{}'.format(
                '42'), 'state': 'ggggggggggggggggggg'}))
        dump(additional)
        tree = ET.ElementTree(additional)
        tree.write(os.path.join(self.current_Env_path, self.file_name+'.add.xml'),
                   pretty_print=True, encoding='UTF-8', xml_declaration=True)

    def test_net(self):
        self.generate_cfg(False)

        os.system('sumo-gui -c {}.sumocfg'.format(os.path.join(self.current_Env_path,
                                                               self.file_name+'_simulate')))

    def sumo_gui(self):
        self.generate_cfg(True)
        os.system('sumo-gui -c {}.sumocfg '.format(
            os.path.join(self.current_Env_path, self.file_name+'_simulate')))


if __name__ == '__main__':
    network = Network(configs)
    network.sumo_gui()

# class TrafficLight():
#     def __init__(self,):


# node_id = ["n_nl", "n_nc", "n_nr",
#            "n_ln", "n_cnl", "n_cnc", "n_cnr", "n_rn",
#            "n_lc", "n_ccl", "n_ccc", "n_ccr", "n_rc",
#            "n_ls", "n_csl", "n_csc", "n_csr", "n_rs",
#            "n_sl", "n_sc", "n_sr"]

# edge_dict = {
#     "n_nl": ["n_cnl"],
#     "n_nc": ["n_cnc"],
#     "n_nr": ["n_cnr"],
#     "n_ln": ["n_cnl"],
#     "n_lc": ["n_ccl"],
#     "n_ls": ["n_csl"],
#     "n_rn": ["n_cnr"],
#     "n_rc": ["n_ccr"],
#     "n_rs": ["n_csr"],
#     "n_sl": ["n_csl"],
#     "n_sc": ["n_csc"],
#     "n_sr": ["n_csr"],
#     "n_cnl": ["n_nl", "n_ln", "n_cnr", "n_csl"],
#     "n_cnc": ["n_nc", "n_ccc", "n_cnl", "n_cnr"],
#     "n_cnr": ["n_nr", "n_cnl", "n_rn", "n_csr"],
#     "n_csl": ["n_ccl", "n_ls", "n_csc", "n_sl"],
#     "n_csc": ["n_csl", "n_ccc", "n_sc", "n_csr"],
#     "n_csr": ["n_csc", "n_ccr", "n_rs", "n_sr"],
#     "n_ccl": ["n_lc", "n_csl", "n_cnl", "n_ccc"],
#     "n_ccc": ["n_ccl", "n_cnc", "n_ccr", "n_csc"],
#     "n_ccr": ["n_ccc", "n_cnr", "n_rc", "n_csr"],
# }

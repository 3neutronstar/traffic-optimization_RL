import xml.etree.cElementTree as ET
from xml.etree.ElementTree import dump
from lxml import etree as ET
import os
E=ET.Element
Edges=list()
Nodes=list()
Vehicles=list()
configs={
    'num_lanes':3,
    'file_name':'3x3intersection',
    'laneLength':100.0,
    'num_cars':1800,
    'flow_start':0,
    'flow_end':9000,
    'sim_start':0,
    'sim_end':10000,
    'edge_info':Edges,
    'node_info':Nodes,
    'vehicle_info':Vehicles,
}
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
def indent(elem, level=0): 
    i = "\n\t" + level*"" 
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
        self.configs=configs
        self.sim_start=self.configs['sim_start']
        self.sim_end=self.configs['sim_end']
        self.file_name=self.configs['file_name']
        self.xml_node_name = self.configs['file_name']+'.nod'
        self.xml_edg_name = self.configs['file_name']+'.edg'
        self.xml_route_pos= self.file_name+'.rou'
        self.current_path = './'
        self.num_cars=str(self.configs['num_cars'])
        self.num_lanes = str(self.configs['num_lanes'])
        self.flow_start= str(self.configs['flow_start'])
        self.flow_end=str(self.configs['flow_end'])
        self.laneLength=self.configs['laneLength']
        self.nodes=list()
        self.flows = list()
        self.vehicles=list()
        self.edges=list()
        self.connections=list()


    def specify_edge(self):
        edges=list()
        '''
        상속을 위한 함수
        '''
        return edges

    def specify_node(self):
        nodes=list()
        '''
        상속을 위한 함수
        '''

        return nodes

    def _generate_nod_xml(self):
        self.nodes=self.specify_node()
        self.xml_nod_pos = os.path.join(self.current_path, self.xml_node_name)
        nod_xml = ET.Element('nodes')
        
        for node_dict in self.nodes:
            # node_dict['x']=format(node_dict['x'],'.1f')
            nod_xml.append(E('node',attrib=node_dict))
            indent(nod_xml)
        '''
        for node_attributes in nodes:
            x.append(E('node', **node_attributes))
        '''
        dump(nod_xml)
        tree=ET.ElementTree(nod_xml)
        #tree.write(self.xml_nod_pos+'.xml',encoding='utf-8',xml_declaration=True)
        tree.write(self.xml_nod_pos+'.xml', pretty_print=True, encoding='UTF-8', xml_declaration=True)

    def _generate_edg_xml(self):
        self.edges=self.specify_edge()
        self.xml_edg_pos = os.path.join(self.current_path, self.xml_edg_name)
        edg_xml = ET.Element('edges')
        for _, edge_dict in enumerate(self.edges):
            edg_xml.append(E('edge',attrib=edge_dict))
            indent(edg_xml)
        dump(edg_xml)
        tree=ET.ElementTree(edg_xml)
        #tree.write(self.xml_edg_pos+'.xml',encoding='utf-8',xml_declaration=True)
        tree.write(self.xml_edg_pos+'.xml', pretty_print=True, encoding='UTF-8', xml_declaration=True)
    
    def _generate_net_xml(self):
        if len(self.connections)==0:
            os.system('netconvert -n {}.nod.xml -e {}.edg.xml -o {}.net.xml'.format(self.file_name,self.file_name,self.file_name))
        else: #connection이 존재하는 경우 -x

            os.system('netconvert -n {}.nod.xml -e {}.edg.xml -x {}.con.xml -o {}.net.xml'.format(self.file_name,self.file_name,self.file_name,self.file_name))


    def specify_flow(self):
        flows=list()

        return flows

    def specify_connection(self):
        connections=list()
        return connections

    
    def _generate_rou_xml(self):
        self.specify_flow()
        route_xml = ET.Element('routes')
        if len(self.vehicles)!=0: #empty
            for _,vehicle_dict in enumerate(self.vehicles):
                route_xml.append(E('veh',attrib=vehicle_dict))
                indent(route_xml)
        if len(self.flows)!=0:
            for _,flow_dict in enumerate(self.flows):
                route_xml.append(E('flow',attrib=flow_dict))
                indent(route_xml)
        dump(route_xml)
        tree=ET.ElementTree(route_xml)
        tree.write(self.xml_route_pos+'.xml', pretty_print=True, encoding='UTF-8', xml_declaration=True)

    def _generate_cfg(self,route):
        sumocfg=ET.Element('configuration')
        inputXML=ET.SubElement(sumocfg,'input')
        inputXML.append(E('net-file',attrib={'value':self.file_name+'.net.xml'}))
        indent(sumocfg)
        if route==True:
            if os.path.exists(os.path.join(self.current_path,self.file_name+'.rou.xml')):
                inputXML.append(E('route-files',attrib={'value':self.file_name+'.rou.xml'}))
                indent(sumocfg)

        if os.path.exists(os.path.join(self.current_path,self.file_name+'.add.xml')):
            inputXML.append(E('add-file',attrib={'value':self.file_name+'.add.xml'}))
            indent(sumocfg)

        time=ET.SubElement(sumocfg,'time')
        time.append(E('begin',attrib={'value':str(self.sim_start)}))
        indent(sumocfg)
        time.append(E('end',attrib={'value':str(self.sim_end)}))
        indent(sumocfg)
        outputXML=ET.SubElement(sumocfg,'output')
        outputXML.append(E('netstate-dump',attrib={'value':self.file_name+'_dump.net.xml'}))
        indent(sumocfg)
        dump(sumocfg)
        tree=ET.ElementTree(sumocfg)
        tree.write(self.file_name+'_simulate'+'.sumocfg',pretty_print=True,encoding='UTF-8',xml_declaration=True)

    def _generate_add_xml(self):
        additional=ET.Element('additional')
        additional.append(E('edgeData',attrib=))
        indent(sumocfg)

    def test_net(self):
        self._generate_edg_xml()
        self._generate_nod_xml()
        self._generate_net_xml()
        self._generate_cfg(False)
        
        os.system('sumo-gui -c {}.sumocfg'.format(self.file_name+'_simulate'))

    def sumo_gui(self):
        self._generate_edg_xml()
        self._generate_nod_xml()
        self._generate_net_xml()
        self._generate_rou_xml()
        self._generate_cfg(True)
        os.system('sumo-gui -c {}.sumocfg '.format(self.file_name+'_simulate'))



if __name__ == '__main__':
    network = Network(configs)
    network.sumo_gui()

    

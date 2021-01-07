import xml.etree.cElementTree as ET
from xml.etree.ElementTree import dump
from lxml import etree as ET
import os
E=ET.Element

configs={
    'num_lanes':3,
    'file_name':'3x3intersection',
    'laneLength':100.0,
    'num_cars':1800,
    'flow_start':0,
    'flow_end':9000,
    'sim_start':0,
    'sim_end':10000,
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
    def __init__(self, node_id, edge_dict, configs):
        self.sim_start=configs['sim_start']
        self.sim_end=configs['sim_end']
        self.node_id = node_id
        self.edge_dict = edge_dict
        self.file_name=configs['file_name']
        self.xml_node_name = configs['file_name']+'.nod'
        self.xml_edg_name = configs['file_name']+'.edg'
        self.xml_route_pos= self.file_name+'.rou'
        self.current_path = './'
        self.num_cars=str(configs['num_cars'])
        self.num_lanes = str(configs['num_lanes'])
        self.flow_start= str(configs['flow_start'])
        self.flow_end=str(configs['flow_end'])
        self.laneLength=configs['laneLength']
        print(configs)
        self.node = self.specify_node()  # node info


    def specify_edge(self):
        edges = list()
        edges_dict=dict()
        for _, dict_key in enumerate(self.edge_dict.keys()):
            for i, _ in enumerate(self.edge_dict[dict_key]):
                print('    <edge from="{}" id="{}_to_{}" to="{}" numLanes="{}"/>'.format(dict_key,
                                                                                         dict_key,
                                                                                         self.edge_dict[dict_key][i],
                                                                                         self.edge_dict[dict_key][i],
                                                                                         self.num_lanes))
                edges_dict={
                    'from':dict_key,
                    'id':"{}_to_{}".format(dict_key, self.edge_dict[dict_key][i]),
                    'to': self.edge_dict[dict_key][i]
                }
                edges.append(edges_dict)
                edges_dict=dict()
        return edges

    def specify_node(self):
        node_list = list()
        node_dict = dict()
        for _,node in enumerate(self.node_id):
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
            node_dict['x'] = str('%.1f' %x)
            node_dict['y'] = str('%.1f' %y)
            node_list.append(node_dict)
            node_dict=dict()
        self.node = node_list
        return node_list

    def generate_nod_xml(self):
        self.specify_node()
        self.xml_nod_pos = os.path.join(self.current_path, self.xml_node_name)
        nod_xml = ET.Element('nodes')
        
        for node_dict in self.node:
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

    def generate_edg_xml(self):
        self.xml_edg_pos = os.path.join(self.current_path, self.xml_edg_name)
        edg_xml = ET.Element('edges')
        for _, dict_key in enumerate(self.edge_dict.keys()):
            for i, _ in enumerate(self.edge_dict[dict_key]):
                doc = ET.SubElement(edg_xml, 'edge')
                doc.attrib['from']=dict_key
                doc.attrib['id']='{}_to_{}'.format(dict_key,edge_dict[dict_key][i])
                doc.attrib['to']=edge_dict[dict_key][i]
                doc.attrib['numLanes']=self.num_lanes
                edg_xml.append(doc)
                indent(edg_xml)
                #print(print_str)
        dump(edg_xml)
        tree=ET.ElementTree(edg_xml)
        #tree.write(self.xml_edg_pos+'.xml',encoding='utf-8',xml_declaration=True)
        tree.write(self.xml_edg_pos+'.xml', pretty_print=True, encoding='UTF-8', xml_declaration=True)
    
    def generate_net_xml(self):
        os.system('netconvert --node-files={}.nod.xml --edge-files={}.edg.xml --output-file={}.net.xml'.format(self.file_name,self.file_name,self.file_name))

    def specify_flow(self):
        edge_start = list()
        node_fin = list()
        flows = list()
        flow_dict=dict()
        edge_fin = list()
        route_xml = ET.Element('routes')
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
                flow_dict={'id':key_start,'from':append_start_str,'to':append_fin_str,'begin':self.flow_start,'end':self.flow_end,'number':self.num_cars}
                flows.append(flow_dict)
                route_xml.append(E('flow',attrib=flow_dict))
                flow_dict=dict()
                indent(route_xml)
                print('    <flow id="{}" from="{}" to="{}" begin="0" end="9000" number="1800" />'.format(
                    key_start, append_start_str, append_fin_str))
        dump(route_xml)
        tree=ET.ElementTree(route_xml)
        #tree.write(self.xml_edg_pos+'.xml',encoding='utf-8',xml_declaration=True)
        tree.write(self.xml_route_pos+'.xml', pretty_print=True, encoding='UTF-8', xml_declaration=True)
        return flows
    
    def generate_cfg(self):
        sumocfg=ET.Element('configuration')
        input=ET.SubElement(sumocfg,'input')
        input.append(E('net-file',attrib={'value':self.file_name+'.net.xml'}))
        indent(sumocfg)
        input.append(E('route-files',attrib={'value':self.file_name+'.rou.xml'}))
        indent(sumocfg)
        if os.path.exists(os.path.join(self.current_path,self.file_name+'.add.xml')):
            input.append(E('add-file',attrib={'value':self.file_name+'.add.xml'}))
        time=ET.SubElement(sumocfg,'time')
        time.append(E('begin',attrib={'value':str(self.sim_start)}))
        indent(sumocfg)
        time.append(E('end',attrib={'value':str(self.sim_end)}))
        indent(sumocfg)
        dump(sumocfg)
        tree=ET.ElementTree(sumocfg)
        tree.write(self.file_name+'.sumocfg',pretty_print=True,encoding='UTF-8',xml_declaration=True)

    
    def sumogui(self):
        os.system('sumo-gui -c {}.sumocfg'.format(self.file_name))




if __name__ == '__main__':
    network = Network(node_id, edge_dict, configs)
    network.generate_edg_xml()
    network.generate_nod_xml()
    network.generate_net_xml()
    network.specify_flow()
    network.generate_cfg()
    network.sumogui()

    

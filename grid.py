from gen_net import Network
from gen_net import configs
import math


class GridNetwork(Network):
    def __init__(self, configs, gridNum):
        super().__init__(configs)
        self.gridNum = gridNum

    def specify_node(self):
        nodes = list()
        # inNode
        #
        #   . .
        #   | |
        # .-*-*-.
        #   | |
        #   . .
        center = float(self.gridNum)/2.0
        for x in range(self.gridNum):
            for y in range(self.gridNum):
                node_info = dict()
                node_info = {
                    'id': 'n_'+str(x)+'_'+str(y),
                    'type': 'traffic_light'
                }
                # if self.gridNum % 2==0: # odd due to index rule
                #     grid_x=self.configs['laneLength']*(x-center_x)
                #     grid_x=self.configs['laneLength']*(center_y-y)

                # else: # even due to index rule
                grid_x = self.configs['laneLength']*(x-center)
                grid_y = self.configs['laneLength']*(center-y)

                node_info['x'] = str('%.1f' % grid_x)
                node_info['y'] = str('%.1f' % grid_y)
                nodes.append(node_info)

        # outNode
        #  * *
        #  | |
        # *-.-.-*
        #  | |
        for i in range(self.gridNum):
            grid_y = (center-i)*self.configs['laneLength']
            grid_x = (i-center)*self.configs['laneLength']
            node_information = [{
                'id': 'n_'+str(i)+'_u',
                'x': str('%.1f' % grid_x),
                'y': str('%.1f' % (-center*self.configs['laneLength']+(self.gridNum+1)*self.configs['laneLength']))
            },
                {
                'id': 'n_'+str(i)+'_r',
                'x': str('%.1f' % (-center*self.configs['laneLength']+(self.gridNum)*self.configs['laneLength'])),
                'y':str('%.1f' % grid_y)
            },
                {
                'id': 'n_'+str(i)+'_d',
                'x': str('%.1f' % grid_x),
                'y': str('%.1f' % (+center*self.configs['laneLength']-(self.gridNum)*self.configs['laneLength']))
            },
                {
                'id': 'n_'+str(i)+'_l',
                'x': str('%.1f' % (+center*self.configs['laneLength']-(self.gridNum+1)*self.configs['laneLength'])),
                'y':str('%.1f' % grid_y)
            }]
            for j in range(len(node_information)):
                nodes.append(node_information[j])
        self.nodes = nodes
        return nodes

    def specify_edge(self):
        edges = list()
        edges_dict = dict()
        for i in range(self.gridNum):
            edges_dict['n_{}_l'.format(i)] = list()
            edges_dict['n_{}_r'.format(i)] = list()
            edges_dict['n_{}_u'.format(i)] = list()
            edges_dict['n_{}_d'.format(i)] = list()

        for y in range(self.gridNum):
            for x in range(self.gridNum):
                edges_dict['n_{}_{}'.format(x, y)] = list()
                if x == 0:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_l'.format(y))
                    edges_dict['n_{}_l'.format(y)].append(
                        'n_{}_{}'.format(x, y))
                if y == 0:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_u'.format(x))
                    edges_dict['n_{}_u'.format(x)].append(
                        'n_{}_{}'.format(x, y))
                if y == self.gridNum-1:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_d'.format(x))
                    edges_dict['n_{}_d'.format(x)].append(
                        'n_{}_{}'.format(x, y))
                if x == self.gridNum-1:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_r'.format(y))
                    edges_dict['n_{}_r'.format(y)].append(
                        'n_{}_{}'.format(x, y))

                if x+1 < self.gridNum:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_{}'.format(x+1, y))

                if y+1 < self.gridNum:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_{}'.format(x, y+1))
                if x-1 >= 0:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_{}'.format(x-1, y))
                if y-1 >= 0:
                    edges_dict['n_{}_{}'.format(x, y)].append(
                        'n_{}_{}'.format(x, y-1))

        for _, dict_key in enumerate(edges_dict.keys()):
            for i, _ in enumerate(edges_dict[dict_key]):
                edge_info = dict()
                edge_info = {
                    'from': dict_key,
                    'id': "{}_to_{}".format(dict_key, edges_dict[dict_key][i]),
                    'to': edges_dict[dict_key][i],
                    'numLanes': self.num_lanes
                }
                edges.append(edge_info)
        print(edges)
        self.edges = edges
        return edges

    def specify_flow(self):
        flows = list()
        self.flows = flows
        return flows

    def specify_connection(self):
        connections = list()
        self.connections = connections
        return connections


configs['file_name'] = 'test_grid'
grid_num = 4
a = GridNetwork(configs, grid_num)
a.test_net()

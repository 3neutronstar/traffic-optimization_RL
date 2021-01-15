from Agent.base import RLAlgorithm
import torch
from torch import nn
import torch.nn.functional as f
import numpy as np
class QNetwork(nn.Module):
    def __init__(self,configs):
        super(QNetwork,self).__init__()
        self.configs=configs
        self.input_size=self.configs['input_size']
        self.output_size=self.configs['output_size']
        self.configs['fc_net']=[40,30]

        #build nn
        self.fc=list()
        for i, layers in enumerate(self.configs['fc_net']):
            if i==1:
                self.fc.append(nn.Linear(self.input_size,layers))
            elif i==len(self.configs['fc_net']):
                self.fc.append(nn.Linear(before_layers,layers))
                self.fc.append(nn.Linear(layers,self.output_size))
            else:
                self.fc.append(nn.Linear(before_layers,layers))
            before_layers=layers

    def forward(self,state):
        x=state
        for _,fc in enumerate(self.fc):
            x=f.relu(fc(x))
        return x # q value




class Trainer(RLAlgorithm):
    def __init__(self,configs):
        super().__init__(configs)
        self.configs=configs
        self.input_size=self.configs['input_size']
        self.output_size=self.configs['output_size']
        self.action_space=self.configs['action_space']
        self.mainQNetwork=QNetwork(self.configs)
        self.targetQNetwork=QNetwork(self.configs)
        self.epsilon=0.5

    def get_action(self,state):
        self.Q=self.mainQNetwork(state)
        self.targetQ=self.mainQNetwork(state)
        _, action=torch.max(self.Q,dim=1)
        action=action.data[0].item() 
        # fc net 거쳐왔으므로 dimension이 1임 그러면 max를 거치면 하나의 요소만 나옴 but list라서 0으로 뽑아내는것

        if np.random.rand(1)<self.epsilon:
            return action
        else:
            return action
    
    def target_update(self):
        self.targetQNetwork.load_state_dict(self.mainQNetwork.state_dict())





        







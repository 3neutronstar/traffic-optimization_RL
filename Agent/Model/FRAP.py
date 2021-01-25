import torch
import torch.nn as nn
import torch.nn.functional as F
class FRAP(nn.Module):
    def __init__(self,input_size,output_size):
        #처음 있는 fc에 input size 정해줄것
        #마지막에 있는 fc에 output size 정해줄것
        phase_input_size=1
        vehicle_input_size=1
        super(FRAP,self).__init__()
        # A
        self.phase_competition_mask=torch.tensor([
        [0,8,8,9,9,9,9,9],
        [8,0,9,8,9,9,9,9],
        [8,9,0,8,9,9,9,9],
        [9,8,8,0,9,9,9,9],#d
        [9,9,9,9,0,8,8,9],
        [9,9,9,9,8,0,9,8],
        [9,9,9,9,8,9,0,8],
        [9,9,9,9,9,8,8,0]])
        self.demand_model_phase=nn.Sequential(
            nn.Linear(phase_input_size,2),
            nn.ReLU(),
        )
        self.demand_model_vehicle=nn.Sequential(
            nn.Linear(phase_input_size,2),
            nn.ReLU(),
            nn.Linear(2,4)
        )
        self.embedding=nn.Sequential(
            nn.Linear(8,16),
            nn.ReLU()
        )
        self.conv_pair=nn.Sequential(
            nn.Conv2d(32,20,kernel_size=(1,1))
        )


    def forward(self,x):
        demand=list()
        phase_demand=list()
        for i in range(8):
            x_phase=self.demand_model_vehicle(x[i])
            x_vehicle=self.demand_model_phase(x[8+i])
            x=torch.cat((x_phase,x_vehicle),dim=1)
            x=self.embedding(x)
            demand.append(x)
        phase_demand.append(torch.sum(demand[0],demand[4]))# a
        phase_demand.append(torch.sum(demand[0],demand[1]))# b
        phase_demand.append(torch.sum(demand[4],demand[5]))#c
        phase_demand.append(torch.sum(demand[1],demand[5]))#d
        phase_demand.append(torch.sum(demand[2],demand[6]))#e
        phase_demand.append(torch.sum(demand[2],demand[3]))#f
        phase_demand.append(torch.sum(demand[6],demand[7]))#g
        phase_demand.append(torch.sum(demand[3],demand[7]))#h
        # phase pair representation
        x=torch.zeros((8,7,32),dtype=torch.float)
        for i,phase_i in enumerate(phase_demand):
            for j,phase_j in enumerate(phase_demand):
                x[i,j]=torch.cat(phase_i,phase_j,dim=1) # 
        x=self.conv_pair(x)

        # phase competition mask

        
        
        
        

        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
class FRAP(nn.Module):
    def __init__(self,input_size,output_size):
        #처음 있는 fc에 input size 정해줄것
        #마지막에 있는 fc에 output size 정해줄것
        super(FRAP,self).__init__()
        self.conv=nn.Conv2d(input_size,2,kernel_size=(2,2)) 
        self.conv2=nn.Conv2d(2,3,kernel_size=(2,2))

    def forward(self,state):
        x=state
        x=self.conv(x)
        x=self.conv2(x)

        return x


a=FRAP()
a.add_module('a',nn.Sequential(nn.Linear(16,2), nn.ReLU(),nn.Linear(3,3)))
print(a)
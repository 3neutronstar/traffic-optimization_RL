import torch
from torch import nn


class FRAP(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(self.input_size)
        self.cnn = nn.Conv2d()

    def forward(self, input):
        x = input

        return x

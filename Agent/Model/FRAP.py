import torch
import torch.nn as nn
import torch.nn.functional as F


class FRAP(nn.Module):
    def __init__(self, input_size, output_size):
        # 처음 있는 fc에 input size 정해줄것
        # 마지막에 있는 fc에 output size 정해줄것
        phase_input_size = 1
        vehicle_input_size = 1
        super(FRAP, self).__init__()
        # A
        self.phase_competition_mask = torch.tensor([
            [0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
            [0.5, 0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0],
            [0.5, 1.0, 0, 0.5, 1.0, 1.0, 1.0, 1.0],
            [1.0, 0.5, 0.5, 0, 1.0, 1.0, 1.0, 1.0],  # d
            [1.0, 1.0, 1.0, 1.0, 0, 0.5, 0.5, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.5, 0, 1.0, 0.5],
            [1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 0, 0.5],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0]])  # 완전 겹치면 1, 겹치다 말면 0.5 자기자신은 0
        self.demand_model_phase = nn.Sequential(
            nn.Linear(phase_input_size, 2),
            nn.ReLU(),
            nn.Linear(2, 4),
            nn.ReLU(),
        )
        self.demand_model_vehicle = nn.Sequential(
            nn.Linear(vehicle_input_size, 2),
            nn.ReLU(),
            nn.Linear(2, 4),
            nn.ReLU(),
        )
        self.embedding = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU()
        )
        self.conv_pair = nn.Sequential(
            nn.Conv2d(32, 20, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=(1, 1)),
            nn.ReLU(),
        )
        self.conv_mask_pair = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(4, 20, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(20, 20, kernel_size=(1, 1)),
            nn.ReLU(),
        )
        self.conv_competition = nn.Sequential(
            nn.Conv2d(20, 8, kernel_size=(1, 1)),
            nn.Conv2d(8, 1, kernel_size=(1, 1)),
        )

    def forward(self, x):
        demand = list()
        phase_demand = list()
        for i in range(8):
            x_phase = self.demand_model_vehicle(x[0][i].view(1, 1))
            x_vehicle = self.demand_model_phase(x[0][8+i].view(1, 1))
            x = torch.cat((x_phase, x_vehicle), dim=1)
            x = self.embedding(x)
            demand.append(x)
        print(demand[7])
        # element wise sum
        phase_demand.append(torch.add(demand[0], demand[4]))  # a
        phase_demand.append(torch.add(demand[0], demand[1]))  # b
        phase_demand.append(torch.add(demand[4], demand[5]))  # c
        phase_demand.append(torch.add(demand[1], demand[5]))  # d
        phase_demand.append(torch.add(demand[2], demand[6]))  # e
        phase_demand.append(torch.add(demand[2], demand[3]))  # f
        phase_demand.append(torch.add(demand[6], demand[7]))  # g
        phase_demand.append(torch.add(demand[3], demand[7]))  # h
        # phase pair representation
        x = torch.zeros((8, 8, 32), dtype=torch.float)
        for i, phase_i in enumerate(phase_demand):
            for j, phase_j in enumerate(phase_demand):
                x[i, j] = torch.cat((phase_i, phase_j), dim=1)
        print(x)
        x = self.conv_pair(x)

        # phase competition mask
        y = self.conv_mask_pair(self.phase_competition_mask)

        results = torch.mul(x, y)  # element-wise multiplication
        results = self.conv_competition(results)
        results = torch.sum(results, 1)  # size (1,8)

        return results

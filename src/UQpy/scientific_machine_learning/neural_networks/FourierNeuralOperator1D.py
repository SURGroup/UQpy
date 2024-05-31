import torch
import torch.nn as nn
import torch.nn.functional as F
from UQpy.scientific_machine_learning.baseclass import NeuralNetwork
from UQpy.scientific_machine_learning.layers import SpectralConv1d


class FourierNeuralOperator1D(NeuralNetwork):
    def __init__(self, modes: int, width: int, padding: int = 0, **kwargs):
        """

        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)

        :param modes:
        :param width:
        :param kwargs:
        """
        super().__init__(**kwargs)

        self.modes = modes
        self.width = width
        self.padding = padding  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(
            2, self.width
        )  # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 32)
        self.fc2 = nn.Linear(32, 6)

    def forward(self, x):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        # todo: note last FNO does not have activation

        if self.padding > 0:
            x = x[..., : -self.padding]  # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)

        x = self.fc1(x)  # ToDo: User should be allowed to place constraints
        x = F.relu(x)

        x = self.fc2(x)  # ToDo: default linear always applied
        return x

    # def get_grid(self, shape, device):
    #     batchsize, size_x = shape[0], shape[1]
    #     gridx = torch.tensor(torch.linspace(0, 1, size_x), dtype=torch.float)
    #     gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
    #     return gridx.to(device)

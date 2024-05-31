import torch
import torch.nn as nn
import UQpy.scientific_machine_learning as sml
from UQpy.scientific_machine_learning.baseclass import Layer
from typing import Union
from beartype import beartype


@beartype
class FourierBlock1d(Layer):

    def __init__(self, width, modes, activation: Union[None, nn.Module] = nn.ReLU()):
        """

        :param width:
        :param modes:
        """
        super().__init__()
        self.width = width
        self.modes = modes
        self.activation = activation
        self.spectral_conv = sml.SpectralConv1d(self.width, self.width, self.modes)
        self.conv = nn.Conv1d(self.width, self.width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        y1 = self.spectral_conv(x)
        y2 = self.conv(x)
        if self.activation is None:
            return y1 + y2
        else:
            return self.activation(y1 + y2)

    def extra_repr(self) -> str:
        s = f"width={self.width}, modes={self.modes}"
        if self.activation is not nn.ReLU():
            s += f", activation={self.activation}"
        return s

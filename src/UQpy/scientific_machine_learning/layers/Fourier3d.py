import torch
import torch.nn as nn
import UQpy.scientific_machine_learning as sml
from UQpy.scientific_machine_learning.baseclass import Layer
from UQpy.utilities.ValidationTypes import PositiveInteger


class Fourier3d(Layer):

    def __init__(
        self,
        width: PositiveInteger,
        modes1: PositiveInteger,
        modes2: PositiveInteger,
        modes3: PositiveInteger,
        **kwargs,
    ):
        """

        :param width:
        :param modes1:
        :param modes2:
        :param modes3:
        """
        super().__init__(**kwargs)
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.conv = nn.Conv3d(self.width, self.width, (1, 1, 1))
        self.spectral_conv = sml.SpectralConv3d(
            self.width, self.modes1, self.modes2, self.modes3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computational call

        :param x: Tensor of shape
        :return: Tensor of shape
        """
        return self.spectral_conv(x) + self.conv(x)

    def extra_repr(self) -> str:
        return f"width={self.width}, modes1={self.modes1}, modes2={self.modes2}, modes3={self.modes3}"

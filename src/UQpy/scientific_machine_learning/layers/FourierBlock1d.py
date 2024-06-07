import torch
import torch.nn as nn
import UQpy.scientific_machine_learning as sml
from UQpy.scientific_machine_learning.baseclass import Layer
from UQpy.utilities.ValidationTypes import PositiveInteger
from beartype import beartype


@beartype
class FourierBlock1d(Layer):

    def __init__(self, width: PositiveInteger, modes: PositiveInteger):
        """Fourier block for Fourier Neural Operator. Computes ``spectral_convolution(x) + convolution(x)``

        :param width: Number of in and out channels
        :param modes:
        """
        super().__init__()
        self.width = width
        self.modes = modes

        self.spectral_conv = sml.SpectralConv1d(self.width, self.width, self.modes)
        """Spectral convolution defined by ``UQpy.sml.SpectralConv1d(in_channels=width, out_channels=width, modes=modes)"""
        self.conv = nn.Conv1d(self.width, self.width, 1)
        """Convolution layer defined by ``torch.nn.Conv1d(in_channels=width, out_channels=with, kernel_size=1)``"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Input tensor
        :return: Output tensor
        """
        return self.spectral_conv(x) + self.conv(x)

    def extra_repr(self) -> str:
        return f"width={self.width}, modes={self.modes}"


if __name__ == "__main__":
    lay = sml.BayesianLinear(4, 5)
    print(lay.sampling)
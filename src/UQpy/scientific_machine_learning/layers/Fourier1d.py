import torch
import torch.nn as nn
import UQpy.scientific_machine_learning as sml
from UQpy.scientific_machine_learning.baseclass import Layer
from UQpy.utilities.ValidationTypes import PositiveInteger


class Fourier1d(Layer):
    def __init__(self, width: PositiveInteger, modes: PositiveInteger, **kwargs):
        """Construct a Fourier block to compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x))`

        :param width: Number of neurons in the layer
        :param modes: Number of Fourier modes to keep
        """
        super().__init__(**kwargs)
        self.width = width
        self.modes = modes

        self.conv = nn.Conv1d(self.width, self.width, 1)
        self.spectral_conv = sml.SpectralConv1d(self.width, self.width, self.modes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        return self.spectral_conv(x) + self.conv(x)

    def extra_repr(self) -> str:
        return f"width={self.width}, modes={self.modes}"

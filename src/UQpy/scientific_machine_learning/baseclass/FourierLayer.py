import torch
import torch.nn as nn
from UQpy.scientific_machine_learning.baseclass.Layer import Layer
from UQpy.utilities.ValidationTypes import PositiveInteger
from abc import ABC, abstractmethod


class FourierLayer(Layer, ABC):

    def __init__(self, width: PositiveInteger, modes: PositiveInteger, **kwargs):
        """

        :param width: In_channels and out_channels of the spectral and regular convolution
        :param modes: Number of Fourier modes to keep
        :param kwargs: Keyword arguments for ``torch.nn.Module``
        """
        super().__init__(**kwargs)
        self.width = width
        self.modes = modes

    @property
    @abstractmethod
    def spectral_conv(self) -> nn.Module:
        ...

    @property
    @abstractmethod
    def conv(self) -> nn.Module:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Input tensor
        :return: Output tensor
        """
        return self.spectral_conv(x) + self.conv(x)

    def extra_repr(self) -> str:
        return f"width={self.width}, modes={self.modes}"

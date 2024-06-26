import torch
import torch.nn as nn
import torch.nn.functional as F
import UQpy.scientific_machine_learning as sml
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import Layer
from UQpy.utilities.ValidationTypes import PositiveInteger


class Fourier1d(Layer):
    def __init__(
        self,
        width: PositiveInteger,
        modes: PositiveInteger,
        conv_kwargs: dict = {},
        **kwargs,
    ):
        r"""Construct a 1d Fourier block to compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W(x)`

        :param width: Number of neurons in the layer and channels in the spectral convolution
        :param modes: Number of Fourier modes to keep, at most :math:`\lfloor L / 2 \rfloor + 1`
        :param conv_kwargs: Keyword arguments pass to ``torch.nn.functional.conv1d``
        """
        super().__init__(**kwargs)
        self.width = width
        self.modes = modes
        self.conv_kwargs = conv_kwargs

        self.scale: float = 1 / (self.width**2)
        """Normalizing factor for spectral convolution weights"""
        self.weight_spectral_conv: nn.Parameter = nn.Parameter(
            self.scale
            * torch.rand(self.width, self.width, self.modes, dtype=torch.cfloat)
        )
        r"""Weights for the spectral convolution. 
        Tensor of shape :math:`(\text{width}, \text{width}, \text{modes})` with complex entries"""
        kernel_size = (
            self.conv_kwargs["kernel_size"]
            if ("kernel_size" in self.conv_kwargs)
            else 1
        )
        groups = self.conv_kwargs["groups"] if ("groups" in self.conv_kwargs) else 1
        self.weight_conv: nn.Parameter = nn.Parameter(
            torch.empty(self.width, self.width // groups, kernel_size)
        )
        r"""Weights for the convolution. 
        Tensor of shape :math:`(\text{width}, \text{width} // \text{groups}, \text{kernel_size}`"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W`

        :param x: Tensor of shape :math:`(N, \text{width}, L)`
        :return: Tensor of shape :math:`(N, \text{width}, L)`
        """
        return func.spectral_conv1d(
            x, self.weight_spectral_conv, self.width, self.modes
        ) + F.conv1d(x, self.weight_conv, **self.conv_kwargs)

    def extra_repr(self) -> str:
        s = f"width={self.width}, modes={self.modes}"
        if self.conv_kwargs:
            s += f", conv_kwargs={self.conv_kwargs}"
        return s

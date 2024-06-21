import torch
import torch.nn as nn
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import Layer
from UQpy.utilities.ValidationTypes import PositiveInteger


class SpectralConv1d(Layer):
    def __init__(
        self,
        in_channels: PositiveInteger,
        out_channels: PositiveInteger,
        modes: PositiveInteger,
        **kwargs,
    ):
        r"""Applies FFT, linear transform, then inverse FFT.

        :param in_channels: :math:`C_\text{in}`, Number of channels in the input signal
        :param out_channels: :math:`C_\text{out}`, Number of channels in the output signal
        :param modes: Number of Fourier modes to keep, at most :math:`\lfloor L / 2 \rfloor + 1`
        """
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale: float = 1 / (self.in_channels * self.out_channels)
        """Normalizing factor for weights"""
        self.weights: nn.Parameter = nn.Parameter(
            self.scale
            * torch.rand(
                self.in_channels, self.out_channels, self.modes, dtype=torch.cfloat
            )
        )
        r"""Weights of the Fourier modes. 
        
        Tensor of shape :math:`(C_\text{in}, C_\text{out}, \text{modes})` with dtype ``torch.cfloat``"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the 1d spectral convolution of ``x``

        :param x: Tensor of shape :math:`(N, C_\text{in}, L)`
        :return: Tensor of shape :math:`(N, C_\text{out}, L)`
        """
        return func.spectral_conv1d(x, self.weights, self.out_channels, self.modes)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels},"
            f" out_channels={self.out_channels},"
            f" modes={self.modes}"
        )

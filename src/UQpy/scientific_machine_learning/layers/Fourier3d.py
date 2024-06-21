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
        r"""Construct a 3d Fourier block to compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W`

        :param width: Number of neurons in the layer and channels in the spectral convolution
        :param modes1: Number of Fourier modes to keep, at most :math:`\lfloor H / 2 \rfloor + 1`
        :param modes2: Number of Fourier modes to keep, at most :math:`\lfloor W / 2 \rfloor + 1`
        :param modes3: Number of Fourier modes to keep, at most :math:`\lfloor D / 2 \rfloor + 1`
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
        r"""Compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W`

        :param x: Tensor of shape :math:`(N, \text{width}, H, W, D)`
        :return: Tensor of shape :math:`(N, \text{width}, H, W, D)`
        """
        return self.spectral_conv(x) + self.conv(x)

    def extra_repr(self) -> str:
        return f"width={self.width}, modes1={self.modes1}, modes2={self.modes2}, modes3={self.modes3}"

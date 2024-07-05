import torch
import torch.nn as nn
import torch.nn.functional as F
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import Layer
from UQpy.utilities.ValidationTypes import PositiveInteger


class Fourier2d(Layer):

    def __init__(
        self,
        width: PositiveInteger,
        modes: tuple[PositiveInteger, PositiveInteger],
        **kwargs,
    ):
        r"""Construct a 2d Fourier block to compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W(x)`

        :param width: Number of neurons in the layer and channels in the spectral convolution
        :param modes: Tuple of Fourier modes to keep.
         At most :math:`(\lfloor H / 2 \rfloor + 1, \lfloor W / 2 \rfloor + 1)`

        Shape:

        - Input: :math:`(N, \text{width}, H, W)`
        - Output: :math:`(N, \text{width}, H, W)`

        Example:

        >>> h, w = 64, 128
        >>> modes = (h//2 + 1, w//2 + 1)
        >>> width = 5
        >>> f = sml.Fourier2d(width, modes)
        >>> input = torch.rand(2, width, h, w)
        >>> output = f(input)
        """
        super().__init__(**kwargs)
        self.width = width
        self.modes = modes

        self.scale: float = 1 / (self.width**2)
        """Normalizing factor for spectral convolution weights"""
        shape = (self.width, self.width, *self.modes)
        self.weight1_spectral_conv: nn.Parameter = nn.Parameter(
            self.scale * torch.rand(shape, dtype=torch.cfloat)
        )
        r"""First of two weights for the spectral convolution. 
        Tensor of shape :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]})` with complex entries"""
        self.weight2_spectral_conv: nn.Parameter = nn.Parameter(
            self.scale * torch.rand(shape, dtype=torch.cfloat)
        )
        r"""Second of two weights for the spectral convolution. 
        Tensor of shape :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]})` with complex entries"""
        kernel_size = (1, 1)
        self.weight_conv: nn.Parameter = nn.Parameter(
            torch.empty(self.width, self.width, *kernel_size)
        )
        r"""Weights for the convolution. 
        Tensor of shape :math:`(\text{width}, \text{width}, \text{kernel_size[0]}, \text{kernel_size[1]})`"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W`

        :param x: Tensor of shape :math:`(N, \text{width}, H, W)`
        :return: Tensor of shape :math:`(N, \text{width}, H, W)`
        """
        weights = (self.weight1_spectral_conv, self.weight2_spectral_conv)
        return func.spectral_conv2d(x, weights, self.width, self.modes) + F.conv2d(
            x, self.weight_conv
        )

    def extra_repr(self) -> str:
        return f"width={self.width}, modes={self.modes}"

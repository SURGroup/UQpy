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
        device=None,
    ):
        r"""Construct a 2d Fourier block to compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W(x)`

        :param width: Number of neurons in the layer and channels in the spectral convolution
        :param modes: Tuple of Fourier modes to keep.
         At most :math:`(\lfloor H / 2 \rfloor + 1, \lfloor W / 2 \rfloor + 1)`

        Note this class does *not* accept the ``dtype`` argument
        since Fourier layers require real and complex tensors where appropriate.


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
        super().__init__()
        self.width = width
        self.modes = modes

        shape = (self.width, self.width, *self.modes)
        self.weight_spectral_1: nn.Parameter = nn.Parameter(
            torch.empty(shape, dtype=torch.cfloat, device=device)
        )
        r"""The first of two learnable weights for the spectral convolution of shape
        :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]})` with complex entries"""
        self.weight_spectral_2: nn.Parameter = nn.Parameter(
            torch.empty(shape, dtype=torch.cfloat, device=device)
        )
        r"""The second of two learnable weights for the spectral convolution of shape 
        :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]})` with complex entries"""
        kernel_size = (1, 1)
        self.weight_conv: nn.Parameter = nn.Parameter(
            torch.empty(self.width, self.width, *kernel_size, device=device)
        )
        r"""The learnable weights of the convolutions of shape 
        :math:`(\text{width}, \text{width}, \text{kernel_size[0]}, \text{kernel_size[1]})`"""
        self.bias_conv: nn.Parameter = nn.Parameter(
            torch.empty(self.width, device=device)
        )
        r"""The learnable bias of the convolution of shape :math:`(\text{width})`."""

        k = torch.sqrt(1 / torch.tensor(self.width, device=device))
        self.reset_parameters(-k, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W`

        :param x: Tensor of shape :math:`(N, \text{width}, H, W)`
        :return: Tensor of shape :math:`(N, \text{width}, H, W)`
        """
        weights = (self.weight_spectral_1, self.weight_spectral_2)
        return func.spectral_conv2d(x, weights, self.width, self.modes) + F.conv2d(
            x, self.weight_conv, self.bias_conv
        )

    def extra_repr(self) -> str:
        return f"width={self.width}, modes={self.modes}"

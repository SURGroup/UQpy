import torch
import torch.nn as nn
import torch.nn.functional as F
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import Layer
from UQpy.utilities.ValidationTypes import PositiveInteger


class Fourier3d(Layer):

    def __init__(
        self,
        width: PositiveInteger,
        modes: tuple[PositiveInteger, PositiveInteger, PositiveInteger],
        device=None,
    ):
        r"""Construct a 3d Fourier block to compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W(x)`

        :param width: Number of neurons in the layer and channels in the spectral convolution
        :param modes: Tuple of Fourier modes to keep.
         At most :math:`(\lfloor H / 2 \rfloor + 1, \lfloor W / 2 \rfloor + 1, \lfloor D / 2 \rfloor + 1)`

        Note this class does *not* accept the ``dtype`` argument
        since Fourier layers require real and complex tensors where appropriate.

        Shape:

        - Input: :math:`(N, \text{width}, H, W, D)`
        - Output: :math:`(N, \text{width}, H, W, D)`

        Example:

        >>> h, w, d = 64, 128, 256
        >>> modes = (h//2 + 1, w//2 + 1, d//2 + 1)
        >>> width = 5
        >>> f = sml.Fourier3d(width, modes)
        >>> input = torch.rand(2, width, h, w, d)
        >>> output = f(input)
        """
        super().__init__()
        self.width = width
        self.modes = modes

        shape = (self.width, self.width, *self.modes)
        self.weight_spectral_1: nn.Parameter = nn.Parameter(
            torch.empty(shape, dtype=torch.cfloat, device=device)
        )
        r"""The first of four learnable weights for the spectral convolution of shape
        :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]})` 
        with complex entries"""
        self.weight_spectral_2: nn.Parameter = nn.Parameter(
            torch.empty(shape, dtype=torch.cfloat, device=device)
        )
        r"""The second of four learnable weights for the spectral convolution of shape
        :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]})` 
        with complex entries"""
        self.weight_spectral_3: nn.Parameter = nn.Parameter(
            torch.empty(shape, dtype=torch.cfloat, device=device)
        )
        r"""The third of four learnable weights for the spectral convolution of shape
        Tensor of shape :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]})` 
        with complex entries"""
        self.weight_spectral_4: nn.Parameter = nn.Parameter(
            torch.empty(shape, dtype=torch.cfloat, device=device)
        )
        r"""The fourth of four learnable weights for the spectral convolution of shape
        :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]})` 
        with complex entries"""
        kernel_size = (1, 1, 1)
        self.weight_conv: nn.Parameter = nn.Parameter(
            torch.empty(self.width, self.width, *kernel_size, device=device)
        )
        r"""The learnable weights of the convolution of shape 
        :math:`(\text{width}, \text{width}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`"""
        self.bias_conv: nn.Parameter = nn.Parameter(
            torch.empty(self.width, device=device)
        )
        r"""The learnable bias of the convolution of shape :math:`(\text{out_channels})`"""
        k = torch.sqrt(1 / torch.tensor(self.width, device=device))
        self.reset_parameters(-k, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W`

        :param x: Tensor of shape :math:`(N, \text{width}, H, W, D)`
        :return: Tensor of shape :math:`(N, \text{width}, H, W, D)`
        """
        weights = (
            self.weight_spectral_1,
            self.weight_spectral_2,
            self.weight_spectral_3,
            self.weight_spectral_4,
        )
        return func.spectral_conv3d(x, weights, self.width, self.modes) + F.conv3d(
            x, self.weight_conv, self.bias_conv
        )

    def extra_repr(self) -> str:
        return f"width={self.width}, modes={self.modes}"

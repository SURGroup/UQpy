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

        self.scale: float = 1 / (self.width**2)
        """Normalizing factor for spectral convolution weights"""
        shape = (self.width, self.width, *self.modes)
        self.weight1_spectral_conv: nn.Parameter = nn.Parameter(
            self.scale * torch.rand(shape, dtype=torch.cfloat, device=device)
        )
        r"""First of four weights for the spectral convolution.
        Tensor of shape :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]})` 
        with complex entries"""
        self.weight2_spectral_conv: nn.Parameter = nn.Parameter(
            self.scale * torch.rand(shape, dtype=torch.cfloat, device=device)
        )
        r"""Second of four weights for the spectral convolution.
        Tensor of shape :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]})` 
        with complex entries"""
        self.weight3_spectral_conv: nn.Parameter = nn.Parameter(
            self.scale * torch.rand(shape, dtype=torch.cfloat, device=device)
        )
        r"""Third of four weights for the spectral convolution.
        Tensor of shape :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]})` 
        with complex entries"""
        self.weight4_spectral_conv: nn.Parameter = nn.Parameter(
            self.scale * torch.rand(shape, dtype=torch.cfloat, device=device)
        )
        r"""Fourth of four weights for the spectral convolution.
        Tensor of shape :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]})` 
        with complex entries"""
        kernel_size = (1, 1, 1)
        self.weight_conv: nn.Parameter = nn.Parameter(
            torch.empty(self.width, self.width, *kernel_size, device=device)
        )
        r"""Weights for the convolution. 
        Tensor of shape :math:`(\text{width}, \text{width}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W`

        :param x: Tensor of shape :math:`(N, \text{width}, H, W, D)`
        :return: Tensor of shape :math:`(N, \text{width}, H, W, D)`
        """
        weights = (
            self.weight1_spectral_conv,
            self.weight2_spectral_conv,
            self.weight3_spectral_conv,
            self.weight4_spectral_conv,
        )
        return func.spectral_conv3d(x, weights, self.width, self.modes) + F.conv3d(
            x, self.weight_conv
        )

    def extra_repr(self) -> str:
        return f"width={self.width}, modes={self.modes}"

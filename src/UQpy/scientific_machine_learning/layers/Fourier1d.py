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
        device=None,
    ):
        r"""Construct a 1d Fourier block to compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W(x)`

        :param width: Number of neurons in the layer and channels in the spectral convolution
        :param modes: Number of Fourier modes to keep, at most :math:`\lfloor L / 2 \rfloor + 1`

        Note this class does *not* accept the ``dtype`` argument
        since Fourier layers require real and complex tensors where appropriate.

        Shape:

        - Input: :math:`(N, \text{width}, L)`
        - Output: :math:`(N, \text{width}, L)`

        Example:

        >>> length = 128
        >>> modes = (length // 2) + 1
        >>> width = 8
        >>> f = sml.Fourier1d(width, modes)
        >>> input = torch.rand(2, width, length)
        >>> output = f(input)
        """
        super().__init__()
        self.width = width
        self.modes = modes

        self.weight_spectral_conv: nn.Parameter = nn.Parameter(
            torch.empty(
                self.width, self.width, self.modes, dtype=torch.cfloat, device=device
            )
        )
        r"""The learnable weights of the spectral convolution of shape 
        :math:`(\text{width}, \text{width}, \text{modes})` with complex entries
        
        The values of these weights are sampled from
        :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 
        where :math:`k = \frac{1}{\text{width}}`
        """
        kernel_size = 1
        self.weight_conv: nn.Parameter = nn.Parameter(
            torch.empty(self.width, self.width, kernel_size, device=device)
        )
        r"""The learnable weights of the convolution of shape :math:`(\text{width}, \text{width}, \text{kernel_size})`.
        
        The values of these weights are sampled from
        :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 
        where :math:`k = \frac{1}{\text{width}}`"""
        self.bias_conv: nn.Parameter = nn.Parameter(
            torch.empty(self.width, self.width, kernel_size, device=device)
        )
        r"""The learnable bias of the convolution of shape :math:`(\text{width})`.
        
        If ``bias`` is ``True``, then the values of these biases are sampled from
        :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` 
        where :math:`k = \frac{1}{\text{width}}`
        """

        k = torch.sqrt(1 / width)
        self.reset_parameters(-k, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W(x)`

        :param x: Tensor of shape :math:`(N, \text{width}, L)`
        :return: Tensor of shape :math:`(N, \text{width}, L)`
        """
        return func.spectral_conv1d(
            x, self.weight_spectral_conv, self.width, self.modes
        ) + F.conv1d(x, self.weight_conv, self.bias_conv)

    def extra_repr(self) -> str:
        return f"width={self.width}, modes={self.modes}"

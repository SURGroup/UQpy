import torch
import torch.nn as nn
import torch.nn.functional as F
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import Layer
from UQpy.utilities.ValidationTypes import PositiveInteger
from typing import Union


class Fourier3d(Layer):

    def __init__(
        self,
        width: PositiveInteger,
        modes: tuple[PositiveInteger, PositiveInteger, PositiveInteger],
        bias: bool = True,
        device: Union[torch.device, str] = None,
    ):
        r"""A 3d Fourier layer to compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W(x)`

        :param width: Number of neurons in the layer and channels in the spectral convolution
        :param modes: Tuple of Fourier modes to keep.
         At most :math:`(\lfloor D / 2 \rfloor + 1, \lfloor H / 2 \rfloor + 1, \lfloor W / 2 \rfloor + 1)`
        :param bias: If ``True``, adds a learnable bias to the convolution. Default: ``True``

        Shape:

        - Input: :math:`(N, \text{width}, D, H, W)`
        - Output: :math:`(N, \text{width}, D, H, W)`

        Attributes:

        - **weight_spectral** (:py:class:`torch.nn.Parameter`): The learnable weights for the spectral
          convolution of shape :math:`(4, \text{width}, \text{width}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]})`
          with complex entries. The initial values of these weights are sampled from
          :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where :math:`k = \frac{1}{\text{width}}`.
        - **weight_conv** (:py:class:`torch.nn.Parameter`): The learnable weights of the convolution of shape
          :math:`(\text{width}, \text{width}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`
          with real entries. The :math:`\text{kernel_size} = (1, 1, 1)`. The initial values of these weights are sampled
          from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where :math:`k = \frac{1}{\text{width}}`.
        - **bias_conv** (:py:class:`torch.nn.Parameter`): The learnable bias of the convolution of shape
          :math:`(\text{width})` with real entries. If ``bias`` is ``True``, then the initial values of these weights
          are sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where :math:`k = \frac{1}{\text{width}}`.

        The kernel for the convolution is fixed as :math:`\text{kernel_size}=(1, 1, 1)`.

        Example:

        >>> d, h, w = 64, 128, 256
        >>> modes = (d//2 + 1, h//2 + 1, w//2 + 1)
        >>> width = 5
        >>> f = sml.Fourier3d(width, modes)
        >>> input = torch.rand(2, width, d, h, w)
        >>> output = f(input)
        """
        super().__init__()
        self.width = width
        self.modes = modes
        self.bias = bias

        self.weight_spectral: nn.Parameter = nn.Parameter(
            torch.empty(
                4, self.width, self.width, *self.modes, dtype=torch.float, device=device
            )
        )
        kernel_size = (1, 1, 1)
        self.weight_conv: nn.Parameter = nn.Parameter(
            torch.empty(self.width, self.width, *kernel_size, device=device)
        )
        if self.bias:
            self.bias_conv: nn.Parameter = nn.Parameter(
                torch.empty(self.width, device=device)
            )
        else:
            self.register_parameter("bias_conv", None)
        k = torch.sqrt(1 / torch.tensor(self.width, device=device))
        self.reset_parameters(-k, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W(x)`

        :param x: Tensor of shape :math:`(N, \text{width}, D, H, W)`
        :return: Tensor of shape :math:`(N, \text{width}, D, H, W)`
        """
        weights = self.weight_spectral.to(torch.cfloat)
        return func.spectral_conv3d(x, weights, self.width, self.modes) + F.conv3d(
            x, self.weight_conv, self.bias_conv
        )

    def extra_repr(self) -> str:
        s = "width={width}, modes={modes}"
        if self.bias is False:
            s += ", bias={bias}"
        return s.format(**self.__dict__)

import torch
import torch.nn as nn
import torch.nn.functional as F
import UQpy.scientific_machine_learning as sml
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import Layer
from UQpy.utilities.ValidationTypes import PositiveInteger
from typing import Union


class Fourier1d(Layer):
    def __init__(
        self,
        width: PositiveInteger,
        modes: PositiveInteger,
        bias: bool = True,
        device: Union[torch.device, str] = None,
    ):
        r"""A 1d Fourier layer to compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W(x)`

        :param width: Number of neurons in the layer and channels in the spectral convolution
        :param modes: Number of Fourier modes to keep, at most :math:`\lfloor L / 2 \rfloor + 1`
        :param bias: If ``True``, adds a learnable bias to the convolution. Default: ``True``

        .. note::
            This class does *not* accept the ``dtype`` argument
            since Fourier layers require real and complex tensors as described by the attributes.

        Shape:

        - Input: :math:`(N, \text{width}, L)`
        - Output: :math:`(N, \text{width}, L)`

        Attributes:

        - **weight_spectral** (:py:class:`torch.nn.Parameter`): The learnable weights of the spectral convolution of
          shape :math:`(\text{width}, \text{width}, \text{modes})` with complex entries.
          The initial values of these weights are sampled from
          :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where :math:`k = \frac{1}{\text{width}}`.
        - **weight_conv** (:py:class:`torch.nn.Parameter`): The learnable weights of the convolution of shape
          :math:`(\text{width}, \text{width}, \text{kernel_size})` with real entries. The :math:`\text{kernel_size}=1`.
          The initial values of these weights are sampled from
          :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where :math:`k = \frac{1}{\text{width}}`.
        - **bias_conv** (:py:class:`torch.nn.Parameter`): The learnable bias of the convolution of shape
          :math:`(\text{width})` with real entries.
          If ``bias`` is ``True``, then the initial values of these weights are sampled from
          :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where :math:`k = \frac{1}{\text{width}}`.

        The kernel for the convolution is fixed as :math:`\text{kernel_size} = 1`.

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
        self.bias = bias

        self.weight_spectral: nn.Parameter = nn.Parameter(
            torch.empty(
                self.width, self.width, self.modes, dtype=torch.float, device=device
            )
        )
        kernel_size = 1
        self.weight_conv: nn.Parameter = nn.Parameter(
            torch.empty(self.width, self.width, kernel_size, device=device)
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

        :param x: Tensor of shape :math:`(N, \text{width}, L)`
        :return: Tensor of shape :math:`(N, \text{width}, L)`
        """
        return func.spectral_conv1d(
            x, self.weight_spectral.to(torch.cfloat), self.width, self.modes
        ) + F.conv1d(x, self.weight_conv, self.bias_conv)

    def extra_repr(self) -> str:
        s = "width={width}, modes={modes}"
        if self.bias is False:
            s += ", bias={bias}"
        return s.format(**self.__dict__)

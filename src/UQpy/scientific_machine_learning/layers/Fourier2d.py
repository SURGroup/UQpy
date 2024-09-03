import torch
import torch.nn as nn
import torch.nn.functional as F
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import Layer
from UQpy.utilities.ValidationTypes import PositiveInteger
from typing import Union


class Fourier2d(Layer):

    def __init__(
        self,
        width: PositiveInteger,
        modes: tuple[PositiveInteger, PositiveInteger],
        device: Union[torch.device, str] = None,
    ):
        r"""A 2d Fourier layer to compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W(x)`

        :param width: Number of neurons in the layer and channels in the spectral convolution
        :param modes: Tuple of Fourier modes to keep.
         At most :math:`(\lfloor H / 2 \rfloor + 1, \lfloor W / 2 \rfloor + 1)`

        .. note::
            This class does *not* accept the ``dtype`` argument
            since Fourier layers require real and complex tensors as described by the attributes.

        Shape:

        - Input: :math:`(N, \text{width}, H, W)`
        - Output: :math:`(N, \text{width}, H, W)`

        Attributes:

        - **weight_spectral_1** (:py:class:`torch.nn.Parameter`): The first of two learnable weights for the spectral
          convolution of shape :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]})`
          with complex entries.
          The initial values of these weights are sampled from
          :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where :math:`k = \frac{1}{\text{width}}`.
        - **weight_spectral_2** (:py:class:`torch.nn.Parameter`): The second of two learnable weights for the spectral
          convolution of shape :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]})`
          with complex entries. The initial values of these weights are sampled from
          :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where :math:`k = \frac{1}{\text{width}}`.
        - **weight_conv** (:py:class:`torch.nn.Parameter`): The learnable weights of the convolution of shape
          :math:`(\text{width}, \text{width}, \text{kernel_size[0]}, \text{kernel_size[1]})` with real entries.
          The :math:`\text{kernel_size} = (1, 1)`. The initial values of these weights are sampled from
          :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where :math:`k = \frac{1}{\text{width}}`.
        - **bias_conv** (:py:class:`torch.nn.Parameter`): The learnable bias of the convolution of shape
          :math:`(\text{width})` with real entries.
          If ``bias`` is ``True``, then the initial values of these weights are sampled from
          :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where :math:`k = \frac{1}{\text{width}}`.

        The kernel for the convolution is fixed as :math:`\text{kernel_size}=(1, 1)`.

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
        self.weight_spectral_2: nn.Parameter = nn.Parameter(
            torch.empty(shape, dtype=torch.cfloat, device=device)
        )
        kernel_size = (1, 1)
        self.weight_conv: nn.Parameter = nn.Parameter(
            torch.empty(self.width, self.width, *kernel_size, device=device)
        )
        self.bias_conv: nn.Parameter = nn.Parameter(
            torch.empty(self.width, device=device)
        )
        k = torch.sqrt(1 / torch.tensor(self.width, device=device))
        self.reset_parameters(-k, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W(x)`

        :param x: Tensor of shape :math:`(N, \text{width}, H, W)`
        :return: Tensor of shape :math:`(N, \text{width}, H, W)`
        """
        weights = (self.weight_spectral_1, self.weight_spectral_2)
        return func.spectral_conv2d(x, weights, self.width, self.modes) + F.conv2d(
            x, self.weight_conv, self.bias_conv
        )

    def extra_repr(self) -> str:
        return f"width={self.width}, modes={self.modes}"

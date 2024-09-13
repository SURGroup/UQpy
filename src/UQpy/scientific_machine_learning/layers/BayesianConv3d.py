import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
from typing import Union
from UQpy.scientific_machine_learning.baseclass import NormalBayesianLayer
from UQpy.utilities.ValidationTypes import (
    PositiveInteger,
    NonNegativeInteger,
    PositiveFloat,
)


class BayesianConv3d(NormalBayesianLayer):

    def __init__(
        self,
        in_channels: PositiveInteger,
        out_channels: PositiveInteger,
        kernel_size: Union[
            PositiveInteger, tuple[PositiveInteger, PositiveInteger, PositiveInteger]
        ],
        stride: Union[
            PositiveInteger, tuple[PositiveInteger, PositiveInteger, PositiveInteger]
        ] = 1,
        padding: Union[
            str,
            NonNegativeInteger,
            tuple[NonNegativeInteger, NonNegativeInteger, NonNegativeInteger],
        ] = 0,
        dilation: Union[
            PositiveInteger, tuple[PositiveInteger, PositiveInteger, PositiveInteger]
        ] = 1,
        groups: PositiveInteger = 1,
        bias: bool = True,
        sampling: bool = True,
        prior_mu: float = 0.0,
        prior_sigma: PositiveFloat = 0.1,
        posterior_mu_initial: tuple[float, PositiveFloat] = (0.0, 0.1),
        posterior_rho_initial: tuple[float, PositiveFloat] = (-3.0, 0.1),
        device: Union[torch.device, str] = None,
        dtype: torch.dtype = None,
    ):
        r"""Applies a Bayesian 3D convolution over an input signal composed of several input planes.

        :param in_channels: Number of channels in the input image
        :param out_channels: Number of channels produced by the convolution
        :param kernel_size: Size of the convolving kernel
        :param stride: Stride of the convolution. Default: 1
        :param padding: Padding added to all six sides of the input.
         It can be either a string {‘valid’, ‘same’} or a tuple of ints giving the amount of implicit padding applied on both sides. Default: 0
        :param dilation: Spacing between kernel elements. Default: 1
        :param groups: Number of blocked connections from input channels to output channels.
         ``in_channels`` and ``out_channels`` must both be divisible by ``groups``. Default: 1.
        :param bias: If ``True``, adds a learnable bias to the output. Default: ``True``
        :param sampling: If ``True``, sample layer parameters from their respective Gaussian distributions.
         If ``False``, use distribution mean as parameter values. Default: ``True``
        :param prior_mu: Prior mean, :math:`\mu_\text{prior}` of the prior normal distribution.
         Default: 0.0
        :param prior_sigma: Prior standard deviation, :math:`\sigma_\text{prior}`, of the prior normal distribution.
         Default: 0.1
        :param posterior_mu_initial: Mean and standard deviation of the initial posterior distribution for :math:`\mu`.
         The initial posterior is :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.
         Default: (0.0, 0.1)
        :param posterior_rho_initial: Mean and standard deviation of the initial posterior distribution for :math:`\rho`.
         The initial posterior is :math:`\mathcal{N}(\rho_\text{posterior}[0], \rho_\text{posterior}[1])`.
         The standard deviation of the posterior is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to ensure it is positive.
         Default: (-3.0, 0.1)

        .. note::
            This class calls :func:`torch.nn.functional.conv3d` with ``padding_mode='zeros'``.

        Shape:

        - Input: :math:`(N, C_\text{in},D_\text{in}, H_\text{in}, W_\text{in})` or :math:`(C_\text{in},D_\text{in}, H_\text{in}, W_\text{in})`
        - Output: :math:`(N, C_\text{out},D_\text{out}, H_\text{out}, W_\text{out})` or :math:`(C_\text{out},D_\text{out}, H_\text{out}, W_\text{out})`

        where :math:`D_\text{out} = \left\lfloor \frac{D_\text{in} + 2 \times \text{padding[0]} - \text{dilation[0]} \times (\text{kernel\_size[0] - 1}) - 1}{\text{stride[0]}} + 1\right\rfloor`

        :math:`H_\text{out} = \left\lfloor \frac{H_\text{in} + 2 \times \text{padding[0]} - \text{dilation[0]} \times (\text{kernel\_size[0] - 1}) - 1}{\text{stride[0]}} + 1\right\rfloor`

        :math:`W_\text{out} = \left\lfloor \frac{W_\text{in} + 2 \times \text{padding[1]} - \text{dilation[1]} \times (\text{kernel\_size[1] - 1}) - 1}{\text{stride[1]}} + 1\right\rfloor`

        Attributes:

        Unless otherwise noted, all parameters are initialized using the ``priors`` with values
        from :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`

        - **weight_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean of the weights of the module
          of shape :math:`(\text{out_channels}, \frac{\text{in_channels}}{\text{groups}}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`.
        - **weight_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution standard deviation of the weights of the module
          of shape :math:`(\text{out_channels}, \frac{\text{in_channels}}{\text{groups}}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`.
          The standard deviation is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive.
        - **bias_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean of the bias of the module
          of shape :math:`(\text{out_channels})`. If ``bias`` is ``True``, the values are initialized
          from :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.
        - **bias_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution standard deviation of the bias of the module
          of shape :math:`(\text{out_channels})`. The standard deviation is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to
          guarantee it is positive. If ``bias`` is ``True``, the values are initialized
          from :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.

        Example:

        >>> # With cubic kernels and equal stride
        >>> layer = sml.BayesianConv3d(16, 33, 3, stride=2)
        >>> # non-cubic kernels and unequal stride and with padding
        >>> layer = sml.BayesianConv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        >>> input = torch.randn(20, 16, 10, 50, 100)
        >>> layer.sample(False)
        >>> deterministic_output = layer(input)
        >>> layer.sample()
        >>> probabilistic_output = layer(input)
        >>> print(torch.all(deterministic_output == probabilistic_output))
        tensor(False)
        """
        kernel_size = _triple(kernel_size)
        parameter_shapes = {
            "weight": (out_channels, in_channels // groups, *kernel_size),
            "bias": out_channels if bias else None,
        }
        super().__init__(
            parameter_shapes,
            sampling,
            prior_mu,
            prior_sigma,
            posterior_mu_initial,
            posterior_rho_initial,
            device,
            dtype,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = padding if isinstance(padding, str) else _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply :func:`F.conv3d` to ``x`` where the weight and bias are drawn from random variables

        :param x: Tensor of shape :math:`(N, C_\text{in}, D_\text{in}, H_\text{in}, W_\text{in})`
        :return: Tensor of shape :math:`(N, C_\text{out}, D_\text{out}, H_\text{out}, W_\text{out})`
        """
        weight, bias = self.get_bayesian_weights()
        return F.conv3d(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def extra_repr(self) -> str:
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}"
        if self.stride != (1,) * len(tuple(self.stride)):
            s += ", stride={stride}"
        if self.padding != (0,) * len(tuple(self.padding)):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(tuple(self.dilation)):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is False:
            s += ", bias={bias}"
        if self.sampling is False:
            s += ", sampling={sampling}"
        if self.prior_mu != 0.0:
            s += ", prior_mu={prior_mu}"
        if self.prior_sigma != 0.1:
            s += ", prior_sigma={prior_sigma}"
        if self.posterior_mu_initial != (0.0, 0.1):
            s += ", posterior_mu_initial={posterior_mu_initial}"
        if self.posterior_rho_initial != (-3.0, 0.1):
            s += ", posterior_rho_initial={posterior_rho_initial}"
        return s.format(**self.__dict__)

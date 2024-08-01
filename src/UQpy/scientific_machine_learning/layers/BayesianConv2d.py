import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from typing import Union
from UQpy.scientific_machine_learning.baseclass import BayesianLayer
from UQpy.utilities.ValidationTypes import PositiveInteger, NonNegativeInteger


class BayesianConv2d(BayesianLayer):

    def __init__(
        self,
        in_channels: PositiveInteger,
        out_channels: PositiveInteger,
        kernel_size: Union[PositiveInteger, tuple[PositiveInteger, PositiveInteger]],
        stride: Union[PositiveInteger, tuple[PositiveInteger, PositiveInteger]] = 1,
        padding: Union[
            str, NonNegativeInteger, tuple[NonNegativeInteger, NonNegativeInteger]
        ] = 0,
        dilation: Union[PositiveInteger, tuple[PositiveInteger, PositiveInteger]] = 1,
        groups: int = 1,
        bias: bool = True,
        priors: dict = None,
        sampling: bool = True,
        device: Union[torch.device, str] = None,
        dtype: torch.dtype = None,
    ):
        r"""Applies a Bayesian 2D convolution over an input signal composed of several input planes.

        :param in_channels: Number of channels in the input image
        :param out_channels: Number of channels produced by the convolution
        :param kernel_size: Size of the convolving kernel
        :param stride: Stride of the convolution. Default: 1
        :param padding: Padding added to both sides of the input. Default: 0
         It can be a string ``"valid"`` or ``"same"``, an integer,
         or a tuple of integers giving the amount of implicit padding applied on both sides.
        :param dilation: Spacing between kernel elements. Default: 1
        :param groups: Number of blocked connections from input channels to output channels. Default: 1.
         ``in_channels`` and ``out_channels`` must both be divisible by ``groups``.
        :param bias: If ``True``, adds a learnable bias to the output. Default: ``True``
        :param priors: Prior and posterior distribution parameters. The dictionary keys and their default values are:

         - ``priors["prior_mu"]`` = :math:`0`
         - ``priors["prior_sigma"]`` = :math:`0.1`
         - ``priors["posterior_mu_initial"]`` = ``(0, 0.1)``
         - ``priors["posterior_rho_initial"]`` = ``(-3, 0.1)``
        :param sampling: If ``True``, sample layer parameters from their respective Gaussian distributions.
         If ``False``, use distribution mean as parameter values. Default: ``True``

        .. note::
            This class calls ``torch.nn.functional.conv2d`` with ``padding_mode='zeros'``.

        Shape:

        - Input: :math:`(N, C_\text{in}, H_\text{in}, W_\text{in})` or :math:`(C_\text{in}, H_\text{in}, W_\text{in})`
        - Output: :math:`(N, C_\text{out}, H_\text{out}, W_\text{out})` or :math:`(C_\text{out}, H_\text{out}, W_\text{out})`

        where :math:`H_\text{out} = \left\lfloor \frac{H_\text{in} + 2 \times \text{padding[0]} - \text{dilation[0]} \times (\text{kernel\_size[0] - 1}) - 1}{\text{stride[0]}} + 1\right\rfloor`

        :math:`W_\text{out} = \left\lfloor \frac{W_\text{in} + 2 \times \text{padding[1]} - \text{dilation[1]} \times (\text{kernel\_size[1] - 1}) - 1}{\text{stride[1]}} + 1\right\rfloor`

        Attributes:

        Unless otherwise noted, all parameters are initialized using the ``priors`` with values
        from :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`

        - **weight_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean of the weights of the module
           of shape :math:`(\text{out_channels}, \frac{\text{in_channels}}{\text{groups}}, \text{kernel_size[0]}, \text{kernel_size[1]})`.
        - **weight_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution variance of the weights of the module
          of shape :math:`(\text{out_channels}, \frac{\text{in_channels}}{\text{groups}}, \text{kernel_size[0]}, \text{kernel_size[1]})`.
          The variance is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive.
        - **bias_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean of the bias of the module
          of shape :math:`(\text{out_channels})`. If ``bias`` is ``True``, the values are initialized
          from :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.
        - **bias_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution variance of the bias of the module
          of shape :math:`(\text{out_channels})`. The variance is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to
          guarantee it is positive. If ``bias`` is ``True``, the values are initialized
          from :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.


        Example:

        >>> # With square kernels and equal stride
        >>> layer = sml.BayesianConv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> layer = sml.BayesianConv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> layer = sml.BayesianConv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> layer.sample(False)
        >>> deterministic_output = layer(input)
        >>> layer.sample()
        >>> probabilistic_output = layer(input)
        >>> print(torch.all(deterministic_output == probabilistic_output))
        tensor(False)
        """
        kernel_size = _pair(kernel_size)
        parameter_shapes = {
            "weight": (out_channels, in_channels // groups, *kernel_size),
            "bias": out_channels if bias else None,
        }
        super().__init__(parameter_shapes, priors, sampling, device, dtype)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding if isinstance(padding, str) else _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply ``F.conv2d`` to ``x`` where the weight and bias are drawn from random variables

        :param x: Tensor of shape :math:`(N, C_\text{in}, H_\text{in}, W_\text{in})`
        :return: Tensor of shape :math:`(N, C_\text{out}, H_\text{out}, W_\text{out})`
        """
        weight, bias = self.get_bayesian_weights()
        return F.conv2d(
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
        if self.priors:
            s += ", priors={priors}"
        if not self.sampling:
            s += ", sampling={sampling}"
        return s.format(**self.__dict__)

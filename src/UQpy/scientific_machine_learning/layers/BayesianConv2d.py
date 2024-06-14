import torch
import torch.nn.functional as F
from UQpy.scientific_machine_learning.baseclass import BayesianLayer
from UQpy.utilities.ValidationTypes import PositiveInteger, NonNegativeInteger
from typing import Union


class BayesianConv2d(BayesianLayer):

    def __init__(
        self,
        in_channels: PositiveInteger,
        out_channels: PositiveInteger,
        kernel_size: Union[PositiveInteger, tuple[PositiveInteger, PositiveInteger]],
        stride: Union[PositiveInteger, tuple[PositiveInteger, PositiveInteger]] = 1,
        padding: Union[str, NonNegativeInteger, tuple[NonNegativeInteger, NonNegativeInteger]] = 0,
        dilation: Union[PositiveInteger, tuple[PositiveInteger, PositiveInteger]] = 1,
        groups: int = 1,
        bias: bool = True,
        priors: dict = None,
        sampling: bool = True,
        **kwargs,
    ):
        """Applies a Bayesian 2D convolution over an input signal composed of several input planes.

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
        :param bias: If True, adds a learnable bias to the output. Default: True
        :param priors: Prior and posterior distribution parameters. The dictionary keys and their default values are:

         - ``priors["prior_mu"]`` = :math:`0`
         - ``priors["prior_sigma"]`` = :math:`0.1`
         - ``priors["posterior_mu_initial"]`` = ``(0, 0.1)``
         - ``priors["posterior_rho_initial"]`` = ``(-3, 0.1)``
        :param sampling: If ``True``, sample layer parameters from their respective Gaussian distributions.
         If ``False``, use distribution mean as parameter values.
        """
        kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        weight_shape = (out_channels, in_channels, *kernel_size)
        bias_shape = out_channels if bias else None
        super().__init__(weight_shape, bias_shape, priors, sampling, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model evaluation

        :param x: Tensor of shape :math:`(N, C_\text{in}, H, W)`
        :return: Tensor of shape :math:`(N, C_\text{out}, H, W)`
        """
        weight, bias = self.get_weight_bias()
        return F.conv2d(
            x, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def extra_repr(self) -> str:
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if not self.bias:
            s += ", bias=False"
        if self.priors:
            s += ", priors={priors}"
        if not self.sampling:
            s += ", sampling={sampling}"
        return s.format(**self.__dict__)

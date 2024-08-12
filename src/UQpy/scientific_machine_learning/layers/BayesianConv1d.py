import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _single
from UQpy.scientific_machine_learning.baseclass import BayesianLayer
from UQpy.utilities.ValidationTypes import PositiveInteger, NonNegativeInteger
from typing import Union


class BayesianConv1d(BayesianLayer):

    def __init__(
        self,
        in_channels: PositiveInteger,
        out_channels: PositiveInteger,
        kernel_size: PositiveInteger,
        stride: PositiveInteger = 1,
        padding: Union[str, NonNegativeInteger] = 0,
        dilation: PositiveInteger = 1,
        groups: PositiveInteger = 1,
        bias: bool = True,
        priors: dict = None,
        sampling: bool = True,
        device: Union[torch.device, str] = None,
        dtype: torch.dtype = None,
    ):
        r"""Applies a Bayesian 1D convolution over an input signal composed of several input planes.

        :param in_channels: Number of channels in the input image
        :param out_channels: Number of channels produced by the convolution
        :param kernel_size: Size of the convolving kernel
        :param stride: Stride of the convolution. Default: 1
        :param padding: Padding added to both sides of the input. Default: 0
        :param dilation: Spacing between kernel elements. Default: 1
        :param groups: Number of blocked connections from input channels to output channels. Default: 1.
         ``in_channels`` and ``out_channels`` must both be divisible by ``groups``.
        :param bias: If ``True``, adds a learnable bias to the output. Default: ``True``
        :param priors: Prior and posterior distribution parameters.
         The dictionary keys and their default values are:

         - "prior_mu": 0
         - "prior_sigma" : 0.1
         - "posterior_mu_initial": (0.0, 0.1)
         - "posterior_rho_initial": (-3.0, 0.1)
        :param sampling: If ``True``, sample layer parameters from their respective Gaussian distributions.
         If ``False``, use distribution mean as parameter values. Default: ``True``

        .. note::
            This class calls :func:`torch.nn.functional.conv1d` with ``padding_mode='zeros'``.

        Shape:

        - Input: :math:`(C_\text{in}, N, L_\text{in})` or :math:`(C_\text{in}, L_\text{in})`
        - Output: :math:`(C_\text{out}, N, L_\text{out})` or :math:`(C_\text{out}, L_\text{out})`,

        where :math:`L_\text{out}= \left\lfloor \frac{L_\text{in} + 2 \times \text{padding} - \text{dilation} \times (\text{kernel size} - 1) - 1}{\text{stride}} \right\rfloor + 1`

        Attributes:

        Unless otherwise noted, all parameters are initialized using the ``priors`` with values
        from :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.

        - **weight_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean of the weights of the module
          of shape :math:`(\text{out_channels}, \frac{\text{in_channels}}{\text{groups}}, \text{kernel_size})`.
        - **weight_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution variance of the weights of the module
          of shape :math:`(\text{out_channels}, \frac{\text{in_channels}}{\text{groups}}, \text{kernel_size})`.
          The variance is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive.
        - **bias_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean of the bias of the module
          of shape :math:`(\text{out_channels})`. If ``bias`` is ``True``, the values are initialized
          from :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.
        - **bias_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution variance of the bias of the module
          of shape :math:`(\text{out_channels})`.  The variance is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to
          guarantee it is positive. If ``bias`` is ``True``, the values are initialized
          from :math:`\mathcal{N}(\rho_\text{posterior}[0], \rho_\text{posterior}[1])`.

        Example:

        >>> layer = sml.BayesianConv1d(16, 33, 3, stride=2)
        >>> layer.sample(False)
        >>> input = torch.randn(20, 16, 50)
        >>> deterministic_output = layer(input)
        >>> layer.sample()
        >>> probabilistic_output = layer(input)
        >>> print(torch.all(deterministic_output == probabilistic_output))
        tensor(False)
        """
        parameter_shapes = {
            "weight": (out_channels, in_channels, kernel_size),
            "bias": out_channels if bias else None,
        }
        super().__init__(parameter_shapes, priors, sampling, device, dtype)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.dilation = _single(dilation)
        self.groups = groups
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply :func:`F.conv1d` to ``x`` where the weight and bias are drawn from random variables

        :param x: Tensor of shape :math:`(N, C_\text{in}, L)` or :math:`(C_\text{in}, L)`
        :return: Tensor of shape :math:`(N, C_\text{out}, L)` or :math:`(C_\text{out}, L)`
        """
        weight, bias = self.get_bayesian_weights()
        return F.conv1d(
            x, weight, bias, self.stride, self.padding, self.dilation, self.groups
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

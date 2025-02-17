import torch
import torch.nn.functional as F
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import NormalBayesianLayer
from UQpy.utilities.ValidationTypes import PositiveInteger, PositiveFloat
from typing import Union


class BayesianFourier2d(NormalBayesianLayer):
    def __init__(
        self,
        width: PositiveInteger,
        modes: tuple[PositiveInteger, PositiveInteger],
        bias: bool = True,
        sampling: bool = True,
        prior_mu: float = 0.0,
        prior_sigma: PositiveFloat = 0.1,
        posterior_mu_initial: tuple[float, PositiveFloat] = (0.0, 0.1),
        posterior_rho_initial: tuple[float, PositiveFloat] = (-3.0, 0.1),
        device: Union[torch.device, str] = None,
    ):
        r"""A 2d Bayesian Fourier layer as :math:`\mathcal{F}^{-1} ( R (\mathcal{F}x)) + W(x)`
        where :math:`R`, along with the wieghts and bias for :math:`W`, are normal random variables.

        :param width: Number of neurons in the layer and channels in the spectral convolution
        :param modes: Number of Fourier modes to keep,
         at most :math:`(\lfloor H / 2 \rfloor + 1, \lfloor W / 2 \rfloor + 1)`
        :param bias: If ``True``, adds a learnable bias to the convolution. Default: ``True``
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

        Shape:

        - Input: :math:`(N, \text{width}, H, W)`
        - Output: :math:`(N, \text{width}, H, W)`

        Attributes:

        Unless otherwise noted, all parameters are initialized using the ``priors`` with values
        from :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.

        - **weight_spectral_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean for the
          weights of the spectral convolution of shape
          :math:`(2, \text{width}, \text{width}, \text{modes[0]}, \text{modes[1]})` with complex entries.
        - **weight_spectral_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution standard deviation for
          the weights of the spectral convolution of shape
          :math:`(2, \text{width}, \text{width}, \text{modes[0]}, \text{modes[1]})` with complex entries.
          The standard deviation is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive.
        - **weight_conv_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean for the weights
          of the convolution of shape
          :math:`(\text{width}, \text{width}, \text{kernel_size[0]}, \text{kernel_size[1]})` with real entries.
          The :math:`\text{kernel_size} = (1, 1)`.
        - **weight_conv_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution standard deviation for the weights
          of the convolution of shape
          :math:`(\text{width}, \text{width}, \text{kernel_size[0]}, \text{kernel_size[1]})` with real entries.
          The :math:`\text{kernel_size} = (1, 1)`.
          The standard deviation is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive.
        - **bias_conv_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean for the bias
          of the convolution of shape :math:`(\text{width})` with real entires.
          If ``bias`` is ``True``, the values are initialized from
          :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.
        - **bias_conv_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution standard deviation for the bias
          of the convolution of shape :math:`(\text{width})` with real entries. The standard deviation is computed as
          :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive. If ``bias`` is ``True``, the values are
          initialized from :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.

        Example:

        >>> h, w = 32, 64
        >>> modes = (17, 33)
        >>> width = 9
        >>> layer = sml.BayesianFourier2d(width, modes)
        >>> x = torch.randn(1, width, h, w)
        >>> layer.sample(False)
        >>> deterministic_output = layer(x)
        >>> layer.sample()
        >>> probabilistic_output = layer(x)
        >>> print(torch.all(deterministic_output == probabilistic_output))
        tensor(False)
        """
        kernel_size = (1, 1)
        parameter_shapes = {
            "weight_spectral": (2, width, width, *modes),
            "weight_conv": (width, width, *kernel_size),
            "bias_conv": width if bias else None,
        }
        super().__init__(
            parameter_shapes,
            sampling,
            prior_mu,
            prior_sigma,
            posterior_mu_initial,
            posterior_rho_initial,
            dtype=torch.float,
        )
        self.width = width
        self.modes = modes
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W(x)`

        :param x: Tensor of shape :math:`(N, C_\text{in}, H, W)`
        :return: Tensor of shape :math:`(N, C_\text{in}, H, W)`
        """
        weight_spectral, weight_conv, bias_conv = self.get_bayesian_weights()
        weight_spectral = weight_spectral.to(torch.cfloat)
        return func.spectral_conv2d(
            x, weight_spectral, self.width, self.modes
        ) + F.conv2d(x, weight_conv, bias_conv)

    def extra_repr(self):
        s = "width={width}, modes={modes}"
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

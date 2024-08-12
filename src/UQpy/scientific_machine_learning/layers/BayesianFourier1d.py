import torch
import torch.nn.functional as F
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import BayesianLayer
from UQpy.utilities.ValidationTypes import PositiveInteger
from typing import Union


class BayesianFourier1d(BayesianLayer):
    def __init__(
        self,
        width: PositiveInteger,
        modes: PositiveInteger,
        bias: bool = True,
        priors: dict = None,
        sampling: bool = True,
        device: Union[torch.device, str] = None,
    ):
        r"""A 1d Bayesian Fourier layer as :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W(x)`
        where :math:`R`, along with the wieghts and bias for :math:`W`, are normal random variables.

        :param width: Number of neurons in the layer and channels in the spectral convolution
        :param modes: Number of Fourier modes to keep, at most :math:`\lfloor L / 2 \rfloor + 1`
        :param bias: If ``True``, adds a learnable bias to the convolution. Default: ``True``
        :param priors: Prior and posterior distribution parameters.
         The dictionary keys and their default values are:

         - "prior_mu": 0
         - "prior_sigma" : 0.1
         - "posterior_mu_initial": (0.0, 0.1)
         - "posterior_rho_initial": (-3.0, 0.1)
        :param sampling: If ``True``, sample layer parameters from their respective Gaussian distributions.
         If ``False``, use distribution mean as parameter values. Default: ``True``

        Shape:

        - Input: :math:`(N, \text{width}, L)`
        - Output: :math:`(N, \text{width}, L)`

        Attributes:

        Unless otherwise noted, all parameters are initialized using the ``priors`` with values
        from :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.

        - **weight_spectral_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean of the
          weights of the spectral convolution of shape :math:`(\text{width}, \text{width}, \text{modes})`
          with complex entries.
        - **weight_spectral_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution variance of the
          weights of the spectral convolution of shape :math:`(\text{width}, \text{width}, \text{modes})`
          with complex entries.
          The variance is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive.
        - **weight_conv_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean of the
          weights of the convolution of shape :math:`(\text{width}, \text{width}, \text{kernel_size})`.
          The :math:`\text{kernel_size}=1`.
        - **weight_conv_rho** (:py:class:`torch.nn.Parameter`) The learnable distribution variance of the
          weights of the convolution of shape :math:`(\text{width}, \text{width}, \text{kernel_size})`.
          The :math:`\text{kernel_size}=1`.
          The variance is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive.
        - **bias_conv_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean of the
          bias of the convolution of shape :math:`(\text{width})`.
          If ``bias`` is ``True``, the values are initialized from
          :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.
        - **bias_conv_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution variance of the
          bias of the convolution of shape :math:`(\text{width})`.
          The variance is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive.
          If ``bias`` is ``True``, the values are initialized from
          :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.

        Example:

        >>> length = 128
        >>> modes = (length // 2) + 1
        >>> width = 9
        >>> layer = sml.BayesianFourier1d(width, modes)
        >>> layer.sample(False)
        >>> x = torch.randn(2, width, length)
        >>> deterministic_output = layer(x)
        >>> layer.sample(True)
        >>> probabilistic_output = layer(x)
        >>> print(torch.all(deterministic_output == probabilistic_output))
        tensor(False)
        """
        kernel_size = 1
        parameter_shapes = {
            "weight_spectral": (width, width, modes),
            "weight_conv": (width, width, kernel_size),
            "bias_conv": width if bias else None,
        }
        super().__init__(
            parameter_shapes,
            priors,
            sampling,
            device,
            dtype=torch.float,
        )
        self.width = width
        self.modes = modes
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W(x)`

        :param x: Tensor of shape :math:`(N, \text{width}, L)`
        :return: Tensor of shape :math:`(N, \text{width}, L)`
        """
        weight_spectral, weight_conv, bias_conv = self.get_bayesian_weights()
        weight_spectral = weight_spectral.to(torch.cfloat)
        return func.spectral_conv1d(
            x, weight_spectral, self.width, self.modes
        ) + F.conv1d(x, weight_conv, bias_conv)

    def extra_repr(self):
        s = "width={width}, modes={modes}"
        if self.bias is False:
            s += ", bias={bias}"
        if self.priors:
            s += ", priors={priors}"
        if self.sampling is False:
            s += ", sampling={sampling}"
        return s.format(**self.__dict__)

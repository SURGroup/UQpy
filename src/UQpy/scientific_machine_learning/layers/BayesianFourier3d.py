import torch
import torch.nn.functional as F
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import BayesianLayer
from UQpy.utilities.ValidationTypes import PositiveInteger
from typing import Union


class BayesianFourier3d(BayesianLayer):
    def __init__(
        self,
        width: PositiveInteger,
        modes: tuple[PositiveInteger, PositiveInteger, PositiveInteger],
        bias: bool = True,
        priors: dict = None,
        sampling: bool = True,
        device: Union[torch.device, str] = None,
    ):
        r"""A 3d Bayesian Fourier layer as :math:`\mathcal{F}^{-1} ( R (\mathcal{F}x)) + W(x)`
        where :math:`R`, along with the wieghts and bias for :math:`W`, are random variables.

        :param width: Number of neurons in the layer and channels in the spectral convolution
        :param modes: Number of Fourier modes to keep,
         at most :math:`(\lfloor D / 2 \rfloor + 1, \lfloor H / 2 \rfloor + 1, \lfloor W / 2 \rfloor + 1)`
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

        - Input: :math:`(N, \text{width}, D, H, W)`
        - Output: :math:`(N, \text{width}, D, H, W)`

        Attributes:

        Unless otherwise noted, all parameters are initialized using the ``priors`` with values
        from :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.

        - **weight_spectral_1_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean for the
          first of four weights of the spectral convolution of shape
          :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]})` with complex entries.
        - **weight_spectral_1_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution variance for the
          first of four weights of the spectral convolution of shape
          :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]})` with complex entries.
          The variance is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive.
        - **weight_spectral_2_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean for the
          second of four weights of the spectral convolution of shape
          :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]})` with complex entries.
        - **weight_spectral_2_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution variance for the
          second of four weights of the spectral convolution of shape
          :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]})` with complex entries.
          The variance is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive.
        - **weight_spectral_3_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean for the
          third of four weights of the spectral convolution of shape
          :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]})` with complex entries.
        - **weight_spectral_3_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution variance for the
          third of four weights of the spectral convolution of shape
          :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]})` with complex entries.
          The variance is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive.
        - **weight_spectral_4_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean for the
          fourth of four weights of the spectral convolution of shape
          :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]})` with complex entries.
        - **weight_spectral_4_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution variance for the
          fourth of four weights of the spectral convolution of shape
          :math:`(\text{width}, \text{width}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]})` with complex entries.
          The variance is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive.
        - **weight_conv_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean for the weights
          of the convolution of shape
          :math:`(\text{width}, \text{width}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`
          with real entries. The :math:`\text{kernel_size} = (1, 1, 1)`.
        - **weight_conv_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution variance for the weights
          of the convolution of shape
          :math:`(\text{width}, \text{width}, \text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`
          with real entries. The :math:`\text{kernel_size} = (1, 1, 1)`.
          The variance is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive.
        - **bias_conv_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean for the bias
          of the convolution of shape :math:`(\text{width})` with real entires.
          If ``bias`` is ``True``, the values are initialized from
          :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.
        - **bias_conv_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution variance for the bias
          of the convolution of shape :math:`(\text{width})` with real entries. The variance is computed as
          :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive. If ``bias`` is ``True``, the values are
          initialized from :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.

        Example:

        >>> d, h, w = 16, 32, 64
        >>> modes = (9, 17, 33)
        >>> width = 4
        >>> layer = sml.BayesianFourier3d(width, modes)
        >>> x = torch.randn(1, width, d, h, w)
        >>> layer.sample(False)
        >>> deterministic_output = layer(x)
        >>> layer.sample()
        >>> probabilistic_output = layer(x)
        >>> print(torch.all(determinisitc_output == probabilistic_output))
        tensor(False)
        """
        kernel_size = (1, 1, 1)
        parameter_shapes = {
            "weight_spectral_1": (width, width, *modes),
            "weight_spectral_2": (width, width, *modes),
            "weight_spectral_3": (width, width, *modes),
            "weight_spectral_4": (width, width, *modes),
            "weight_conv": (width, width, *kernel_size),
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

        :param x: Tensor of shape :math:`(N, C_\text{in}, D, H, W)`
        :return: Tensor of shape :math:`(N, C_\text{in}, D, H, W)`
        """
        w1, w2, w3, w4, weight_conv, bias_conv = self.get_bayesian_weights()
        spectral_weights = (
            w1.to(torch.cfloat),
            w2.to(torch.cfloat),
            w3.to(torch.cfloat),
            w4.to(torch.cfloat),
        )
        return func.spectral_conv3d(
            x, spectral_weights, self.width, self.modes
        ) + F.conv3d(x, weight_conv, bias_conv)

    def extra_repr(self):
        s = "width={width}, modes={modes}"
        if self.bias is False:
            s += ", bias={bias}"
        if self.priors:
            s += ", priors={priors}"
        if self.sampling is False:
            s += ", sampling={sampling}"
        return s.format(**self.__dict__)

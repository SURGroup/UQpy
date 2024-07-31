import torch
import torch.nn.functional as F
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import BayesianLayer
from UQpy.utilities.ValidationTypes import PositiveInteger
from typing import Union


class BayesianFourier2d(BayesianLayer):
    def __init__(
        self,
        width: PositiveInteger,
        modes: tuple[PositiveInteger, PositiveInteger],
        bias: bool = True,
        priors: dict = None,
        sampling: bool = True,
        device: Union[torch.device, str] = None,
    ):
        r"""Construct a Bayesian Fourier layer as :math:`\mathcal{F}^{-1} ( R (\mathcal{F}x)) + W(x)`
        where :math:`R` and :math:`W` are normal random variables.

        :param width: Number of neurons in the layer and channels in the spectral convolution
        :param modes: Number of Fourier modes to keep,
         at most :math:`(\lfloor H / 2 \rfloor + 1, \lfloor W / 2 \rfloor + 1)`
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

        - Input: :math:`(N, \text{width}, H, W)`
        - Output: :math:`(N, \text{width}, H, W)`

        Attributes:

        - weight_spectral_1_mu (:py:class:`torch.nn.Parameter`)
        - weight_spectral_1_rho (:py:class:`torch.nn.Parameter`)
        - weight_spectral_2_mu (:py:class:`torch.nn.Parameter`)
        - weight_spectral_2_rho (:py:class:`torch.nn.Parameter`)
        - weight_conv_mu (:py:class:`torch.nn.Parameter`)
        - weight_conv_rho (:py:class:`torch.nn.Parameter`)
        - bias_conv_mu (:py:class:`torch.nn.Parameter`)
        - bias_conv_rho (:py:class:`torch.nn.Parameter`)

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
            "weight_spectral_1": (width, width, *modes),
            "weight_spectral_2": (width, width, *modes),
            "weight_conv": (width, width, *kernel_size),
            "bias_conv": width if bias else None,
        }
        super().__init__(
            parameter_shapes,
            priors,
            sampling,
            device,
            dtype=(torch.cfloat, torch.cfloat, torch.float, torch.float),
        )
        self.width = width
        self.modes = modes
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W(x)`

        :param x: Tensor of shape :math:`(N, C_\text{in}, H, W)`
        :return: Tensor of shape :math:`(N, C_\text{in}, H, W)`
        """
        w1, w2, weight_conv, bias_conv = self.get_bayesian_weights()
        return func.spectral_conv2d(x, (w1, w2), self.width, self.modes) + F.conv2d(
            x, weight_conv, bias_conv
        )

    def extra_repr(self):
        s = "width={width}, modes={modes}"
        if self.bias is False:
            s += ", bias={bias}"
        if self.priors:
            s += ", priors={priors}"
        if self.sampling is False:
            s += ", sampling={sampling}"
        return s.format(**self.__dict__)

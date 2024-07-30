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
        r"""Construct a Bayesian Fourier block as :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W`
        where :math:`R` and :math:`W` are a random variables

        :param width: Number of neurons in the layer and channels in the spectral convolution
        :param modes: Number of Fourier modes to keep, at most :math:`\lfloor L / 2 \rfloor + 1`
        :param priors: Prior and posterior distribution parameters. The dictionary keys and their default values are:

         - ``priors["prior_mu"]`` = :math:`0`
         - ``priors["prior_sigma"]`` = :math:`0.1`
         - ``priors["posterior_mu_initial"]`` = ``(0, 0.1)``
         - ``priors["posterior_rho_initial"]`` = ``(-3, 0.1)``
        :param sampling: If ``True``, sample layer parameters from their respective Gaussian distributions.
         If ``False``, use distribution mean as parameter values.


        Shape:

        - Input: :math:`(N, \text{Width}, L)`
        - Output: :math:`(N, \text{Width}, L)`

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
            dtype=(torch.cfloat, torch.float, torch.float),
        )
        self.width = width
        self.modes = modes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W(x)`

        :param x: Tensor of shape :math:`(N, \text{width}, L)`
        :return: Tensor of shape :math:`(N, \text{width}, L)`
        """
        weight_spectral, weight_conv, bias_conv = self.get_bayesian_weights()
        return func.spectral_conv1d(
            x, weight_spectral, self.width, self.modes
        ) + F.conv1d(x, weight_conv, bias_conv)

    def extra_repr(self) -> str:
        return (
            f"width={self.width}, "
            f"modes={self.modes}, "
            f"priors={self.priors}, "
            f"sampling={self.sampling}"
        )

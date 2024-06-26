import torch
import torch.nn.functional as F
import UQpy.scientific_machine_learning.functional as func
import UQpy.scientific_machine_learning as sml
from UQpy.scientific_machine_learning.baseclass import BayesianLayer
from UQpy.utilities.ValidationTypes import PositiveInteger


class BayesianFourier1d(BayesianLayer):
    def __init__(
        self,
        width: PositiveInteger,
        modes: PositiveInteger,
        priors: dict = None,
        sampling: bool = True,
        **kwargs,
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
        """
        weight_shape = (width, width, modes)  # Parameters for func.spectral_conv1d
        kernel_size = 1
        bias_shape = (width, width, kernel_size)  # Parameters for F.conv1d
        super().__init__(
            weight_shape,
            bias_shape,
            priors,
            sampling=sampling,
            **kwargs,
        )
        self.width = width
        self.modes = modes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute :math:`\mathcal{F}^{-1} (R (\mathcal{F}x)) + W`

        :param x: Tensor of shape :math:`(N, \text{width}, L)`
        :return: Tensor of shape :math:`(N, \text{width}, L)`
        """
        weight_spectral_conv, weight_conv = self.get_weight_bias()
        weight_spectral_conv = weight_spectral_conv.to(torch.cfloat)
        return func.spectral_conv1d(
            x, weight_spectral_conv, self.width, modes=self.modes
        ) + F.conv1d(x, weight_conv)

    def extra_repr(self) -> str:
        return f"width={self.width}, modes={self.modes}, priors={self.priors}, sampling={self.sampling}"

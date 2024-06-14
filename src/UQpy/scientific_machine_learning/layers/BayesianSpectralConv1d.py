import torch
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import BayesianLayer
from UQpy.utilities.ValidationTypes import PositiveInteger


class BayesianSpectralConv1d(BayesianLayer):

    def __init__(
        self,
        in_channels: PositiveInteger,
        out_channels: PositiveInteger,
        modes: PositiveInteger,
        priors=None,
        sampling=True,
        **kwargs,
    ):
        weight_shape = (in_channels, out_channels, modes)
        bias_shape = None
        super().__init__(weight_shape, bias_shape, priors, sampling, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale: float = 1 / (self.in_channels * self.out_channels)  # ToDo: Include in priors?
        """Scalar normalizing factor for weights"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute Fourier coefficients up to factor of :math:`e^{\text{-some constant}}`

        :param x: Tensor of shape :math:`(N, C_\text{in}, L)`
        :return: Tensor of shape :math:`(N, C_\text{out}, L)`
        """
        weights, _ = self.get_weight_bias()
        return func.spectral_conv1d(x, weights, self.out_channels, self.modes)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels},"
            f" out_channels={self.out_channels},"
            f" modes={self.modes},"
            f" sampling={self.sampling}"
        )

import torch
import torch.nn as nn
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
        **kwargs
    ):
        weight_shape = (in_channels, out_channels, modes)
        bias_shape = None
        super().__init__(weight_shape, bias_shape, priors, sampling, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale: float = 1 / (self.in_channels * self.out_channels)
        """Scalar normalizing factor for weights"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Fourier coefficients up to factor of :math:`e^{\\text{-some constant}}`

        :param x: Tensor of shape (n_batch, in_channels, n_x)
        :return: Tensor of shape (n_batch, out_channels, n_x)
        """
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
        )  # multiply relevant Fourier modes

        weights, _ = self.get_weight_bias()
        weights *= self.scale
        out_ft[:, :, : self.modes] = self.complex_multiplication(
            x_ft[:, :, : self.modes], weights
        )

        x = torch.fft.irfft(out_ft, n=x.size(-1))  # return to physical space
        return x

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, modes={self.modes}"

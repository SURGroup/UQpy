import torch
import torch.nn as nn
from UQpy.scientific_machine_learning.baseclass import Layer
from UQpy.utilities.ValidationTypes import PositiveInteger


class SpectralConv1d(Layer):
    def __init__(
        self,
        in_channels: PositiveInteger,
        out_channels: PositiveInteger,
        modes: PositiveInteger,
        **kwargs,
    ):
        """1D Truncated Fourier series. Applies FFT, linear transform, then inverse FFT.

        :param in_channels: Number of channels in the input image
        :param out_channels: Number of channels produced by the convolution
        :param modes:  Number of Fourier modes to multiply, at most floor(N/2) + 1
        """
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale: float = 1 / (self.in_channels * self.out_channels)
        """Scalar normalizing factor for weights"""
        self.weights: nn.Parameter = nn.Parameter(
            self.scale
            * torch.rand(self.in_channels, self.out_channels, self.modes, dtype=torch.cfloat)
        )
        """Learnable parameters to be optimized."""

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
        out_ft[:, :, : self.modes] = self.complex_multiplication(
            x_ft[:, :, : self.modes], self.weights
        )

        x = torch.fft.irfft(out_ft, n=x.size(-1))  # return to physical space
        return x

    @staticmethod
    def complex_multiplication(
        input: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Complex multiplication using ``torch.einsum``

        :param input: Tensor of shape (batch_size, in_channel, n_x)
        :param weights: Tensor for shape (in_channel, out_channel, n_x)
        :return: Tensor of shape (batch_size, out_channel, n_x)
        """
        return torch.einsum("bix,iox->box", input, weights)

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, modes={self.modes}"

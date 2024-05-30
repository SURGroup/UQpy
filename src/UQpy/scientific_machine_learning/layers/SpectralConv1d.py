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
        """1D Fourier layer. Applies FFT, linear transform, then Inverse FFT.

        :param in_channels: Number of channels in the input image
        :param out_channels: Number of channels produced by the convolution
        :param modes:  Number of Fourier modes to multiply, at most floor(N/2) + 1
        """
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale: float = 1 / (in_channels * out_channels)
        """Scalar normalizing factor for weights"""
        self.weights: nn.Parameter = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )
        """Learnable parameters to be optimized."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the truncated spectral convolution of ``x``

        :param x: Input tensor
        :return: Output tensor
        """
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)
        # x_ft = torch.rfft(x,1,normalized=True,onesided=False)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes] = self.complex_multiplication(
            x_ft[:, :, : self.modes], self.weights
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

    @staticmethod
    def complex_multiplication(
        input: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Complex multiplication using ``torch.einsum``

        :param input:
        :param weights:
        :return:
        """
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, modes={self.modes}"

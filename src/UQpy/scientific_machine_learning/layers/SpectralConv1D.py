import torch
import torch.nn as nn
from UQpy.scientific_machine_learning.baseclass import Layer


class SpectralConv1d(Layer):
    def __init__(self, in_channels, out_channels, modes, **kwargs):
        """1D Fourier layer. Applies FFT, linear transform, then Inverse FFT.

        :param in_channels:
        :param out_channels:
        :param modes:  Number of Fourier modes to multiply, at most floor(N/2) + 1
        """
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    @staticmethod
    def complex_multiplication(input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication using ``torch.einsum``

        :param input:
        :param weights:
        :return:
        """
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Input tensor
        :return:
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

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, modes={self.modes}"

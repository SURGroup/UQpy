import torch
import torch.nn as nn
from UQpy.scientific_machine_learning.baseclass import Layer
from UQpy.utilities.ValidationTypes import PositiveInteger


class SpectralConv2d(Layer):
    def __init__(
        self,
        in_channels: PositiveInteger,
        out_channels: PositiveInteger,
        modes1: PositiveInteger,
        modes2: PositiveInteger,
        **kwargs,
    ):
        """2D Fourier layer. It does FFT, linear transform, and Inverse FFT.

        :param in_channels: Number of channels in the input signal
        :param out_channels: Number of channels in the output signal
        :param modes1: Number of Fourier modes to multiply, at most :math:`\lfloor H / 2 \rfloor + 1`
        :param modes2: Number of Fourier modes to multiply, at most :math:`\lfloor W / 2 \rfloor + 1`
        """
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale: float = 1 / (in_channels * out_channels)
        """Normalizing factor for weights"""
        shape = (in_channels, out_channels, self.modes1, self.modes2, 2)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(*shape, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(*shape, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computational call

        :param x: Tensor of shape :math:`(N, C_\text{in}, H, W)`
        :return: Tensor of shape :math:`(N, C_\text{out}, H, W)`
        """
        batch_size = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]

        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x, 2, normalized=True, onesided=True)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            (width // 2) + 1,
            2,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.complex_multiplication_2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.complex_multiplication_2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )
        # Return to physical space
        x = torch.fft.irfft(
            out_ft, 2, normalized=True, onesided=True, signal_sizes=(height, width)
        )
        return x

    @staticmethod
    def complex_multiplication_2d(
        input: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """Complex multiplication over a 2d signal

        :param input: Tensor of shape :math:`(N, C_{\text{in}}, H, W)`
        :param weights: Tensor of shape :math:`(C_\text{in}, C_\text{out}, H, W)`
        :return: Tensor of shape :math:`(N, C_\text{out}, H, W)`
        """
        equation = "bixy,ioxy->boxy"
        upper = torch.einsum(equation, input[..., 0], weights[..., 0]) - torch.einsum(
            equation, input[..., 1], weights[..., 1]
        )
        lower = torch.einsum(equation, input[..., 1], weights[..., 0]) + torch.einsum(
            equation, input[..., 0], weights[..., 1]
        )
        return torch.stack([upper, lower], dim=-1)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels},"
            f" out_channels={self.out_channels},"
            f" modes1={self.modes1},"
            f" modes2={self.modes2}"
        )

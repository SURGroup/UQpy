import torch
import torch.nn as nn
import UQpy.scientific_machine_learning.functional as func
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
        :param modes:  Number of Fourier modes to multiply, at most :math:`\\lfloor L / 2 \\rfloor + 1`
        """
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale: float = 1 / (self.in_channels * self.out_channels)
        """Normalizing factor for weights"""
        self.weights: nn.Parameter = nn.Parameter(
            self.scale
            * torch.rand(
                self.in_channels, self.out_channels, self.modes, dtype=torch.cfloat
            )
        )
        """Weights of the Fourier modes. 
        
        Tensor of shape :math:`(C_\\text{in}, C_\\text{out}, \\text{modes})` with dtype``torch.cfloat``"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        return func.spectral_conv1d(x, self.weights)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     """Computational call
    #
    #     :param x: Tensor of shape :math:`(N, C_\\text{in}, L)`
    #     :return: Tensor of shape :math:`(N, C_\\text{out}, L)`
    #     """
    #     batch_size = x.shape[0]
    #     length = x.shape[2]
    #     # Compute Fourier coefficients up to factor of e^(- something constant)
    #     x_ft = torch.fft.rfft(x)
    #
    #     out_ft = torch.zeros(
    #         batch_size,
    #         self.out_channels,
    #         (length // 2) + 1,
    #         dtype=torch.cfloat,
    #     )  # multiply relevant Fourier modes
    #     out_ft[:, :, : self.modes] = self.complex_multiplication(
    #         x_ft[:, :, : self.modes], self.weights
    #     )
    #
    #     x = torch.fft.irfft(out_ft, n=length)  # return to physical space
    #     return x
    #
    # @staticmethod
    # def complex_multiplication(
    #     input: torch.Tensor, weights: torch.Tensor
    # ) -> torch.Tensor:
    #     """Complex multiplication using ``torch.einsum``
    #
    #     :param input: Tensor of shape :math:`(N, C_\text{in}, L)`
    #     :param weights: Tensor for shape :math:`(C_\text{in}, C_\text{out}, L)`
    #     :return: Tensor of shape :math:`(N, C_\text{out}, L)`
    #     """
    #     return torch.einsum("bix,iox->box", input, weights)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels},"
            f" out_channels={self.out_channels},"
            f" modes={self.modes}"
        )

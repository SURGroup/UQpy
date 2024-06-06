import torch
import torch.nn as nn


class SpectralConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        **kwargs,
    ):
        """3D Fourier layer. It does FFT, linear transform, and Inverse FFT.

        :param in_channels:
        :param out_channels:
        :param modes1: Number of Fourier modes to multiply, at most floor(N/2) + 1
        :param modes2:
        :param modes3:
        """
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale: float = 1 / (in_channels * out_channels)
        """Normalizing factor for weights"""
        shape = (in_channels, out_channels, self.modes1, self.modes2, self.modes3)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(*shape, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(*shape, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(*shape, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(*shape, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computational call

        :param x: Tensor of shape :math:`(N, C_\text{in}, H, W, D)`
        :return: Tensor of shape :math:`(N, C_\text{out}, H, W, D)`
        """
        batch_size = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]
        depth = x.shape[4]

        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            height,
            width,
            (depth // 2) + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = (
            self.complex_multiplication_3d(
                x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
            )
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = (
            self.complex_multiplication_3d(
                x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
            )
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = (
            self.complex_multiplication_3d(
                x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
            )
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = (
            self.complex_multiplication_3d(
                x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
            )
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

    @staticmethod
    def complex_multiplication_3d(input, weights):
        """Complex multiplication over a 3d  signal

        :param input: Tensor of shape :math:`(N, C_\text{in}, H, W, D)`
        :param weights: Tensor of shape :math:`(C_\text{in}, C_\text{out}, H, W, D)`
        :return: Tensor of shape :math:`(N, C_\text{out}, H, W, D)`
        """
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels},"
            f" out_channels={self.out_channels},"
            f" modes1={self.modes1},"
            f" modes2={self.modes2},"
            f" modes3={self.modes3}"
        )

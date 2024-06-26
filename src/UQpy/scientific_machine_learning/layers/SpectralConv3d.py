import torch
import torch.nn as nn
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import Layer
from UQpy.utilities.ValidationTypes import PositiveInteger


class SpectralConv3d(Layer):
    def __init__(
        self,
        in_channels: PositiveInteger,
        out_channels: PositiveInteger,
        modes: tuple[PositiveInteger, PositiveInteger, PositiveInteger],
        **kwargs,
    ):
        r"""Applies 3d FFT, linear transform, then inverse 3d FFT.

        :param in_channels: :math:`C_\text{in}`, Number of channels in the input signal
        :param out_channels: :math:`C_\text{out}`, Number of channels in the output signal
        :param modes: Tuple of Fourier modes to keep.
         At most :math:`(\lfloor H / 2 \rfloor + 1, \lfloor W / 2 \rfloor + 1, \lfloor D / 2 \rfloor + 1)`
        """
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        self.scale: float = 1 / (in_channels * out_channels)
        """Normalizing factor for weights"""
        shape = (in_channels, out_channels, *self.modes)
        self.weights1: nn.Parameter = nn.Parameter(
            self.scale * torch.rand(shape, dtype=torch.cfloat)
        )
        r"""First weights of the Fourier modes.

        Tensor of shape :math:`(C_\text{in}, C_\text{out}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]]})` 
        with dtype ``torch.cfloat``"""
        self.weights2: nn.Parameter = nn.Parameter(
            self.scale * torch.rand(shape, dtype=torch.cfloat)
        )
        r"""Second weights of the Fourier modes.

        Tensor of shape :math:`(C_\text{in}, C_\text{out}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]]})` 
        with dtype ``torch.cfloat``"""
        self.weights3: nn.Parameter = nn.Parameter(
            self.scale * torch.rand(shape, dtype=torch.cfloat)
        )
        r"""Third weights of the Fourier modes.

        Tensor of shape :math:`(C_\text{in}, C_\text{out}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]]})` 
        with dtype ``torch.cfloat``"""
        self.weights4: nn.Parameter = nn.Parameter(
            self.scale * torch.rand(shape, dtype=torch.cfloat)
        )
        r"""Fourth weights of the Fourier modes.

        Tensor of shape :math:`(C_\text{in}, C_\text{out}, \text{modes[0]}, \text{modes[1]}, \text{modes[2]]})` 
        with dtype ``torch.cfloat``"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute the 3d spectral convolution of ``x``

        :param x: Tensor of shape :math:`(N, C_\text{in}, H, W, D)`
        :return: Tensor of shape :math:`(N, C_\text{out}, H, W, D)`
        """
        weights = (self.weights1, self.weights2, self.weights3, self.weights4)
        return func.spectral_conv3d(x, weights, self.out_channels, self.modes)

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, out_channels={self.out_channels}, modes={self.modes}"

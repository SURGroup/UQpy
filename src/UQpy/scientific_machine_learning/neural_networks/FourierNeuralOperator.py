import torch
import torch.nn as nn
import torch.nn.functional as F
import UQpy.scientific_machine_learning as sml
from UQpy.scientific_machine_learning.baseclass import NeuralNetwork
from UQpy.utilities.ValidationTypes import PositiveInteger


class FourierNeuralOperator(NeuralNetwork):

    def __init__(
        self,
        modes: int,
        width: int,
        prefix_network: nn.Module,
        suffix_network: nn.Module,
        padding: int = 0,
        n_layers: PositiveInteger = 4,
        **kwargs,
    ):
        """

        :param modes:
        :param width:
        :param prefix_network:
        :param suffix_network:
        :param padding: 0 for periodic domains. Positive for non-periodic domains
        :param n_layers:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.modes = modes
        self.width = width
        self.prefix_network = prefix_network
        self.suffix_network = suffix_network
        self.padding = padding
        self.n_layers = n_layers

        for i in range(n_layers):
            setattr(
                sml.SpectralConv1d(self.width, self.width, self.modes),
                f"spectral_convolution_{i}",
            )
            setattr(nn.Conv1d(self.width, self.width, 1), f"convolution_{i}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Input tensor
        :return: Output tensor
        """
        grid = self.get_grid(x.shape)
        x = torch.cat((x, grid), dim=-1)
        x = self.prefix_network(x)
        x = x.permute(0, 2, 1)
        if self.padding > 0:
            x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic
        for i in range(self.n_layers):
            x = self.fno_block(x, i)
        if self.padding > 0:
            x = x[..., : -self.padding]  # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.suffix_network(x)
        return x

    def fno_block(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Compute the forward pass of the ``i``-th FNO block

        :param x: Input tensor
        :param i: Index of the FNO block
        :return: Output tensor
        """
        spectral_convolution_i = getattr(f"spectral_convolution_{i}")
        convolution_i = getattr(f"convolution_{i}")
        y1 = spectral_convolution_i(x)
        y2 = convolution_i(x)
        return self.activation_function(y1 + y2)

    def get_grid(self, shape: tuple[int, int]):
        """

        :param shape:
        :return:
        """
        batch_size, size_x = shape[0], shape[1]
        grid = torch.linspace(0, 1, size_x, dtype=torch.float)
        grid = grid.reshape(1, size_x, 1).repeat([batch_size, 1, 1])
        return grid

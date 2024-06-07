import torch
import torch.nn as nn
import torch.nn.functional as F
from UQpy.scientific_machine_learning.baseclass import NeuralNetwork


class FourierNeuralOperator(NeuralNetwork):

    def __init__(
        self,
        spectral_network: nn.Module,
        upscale_network: nn.Module = None,
        downscale_network: nn.Module = None,
        padding: int = 0,
        **kwargs,
    ):
        """

        :param spectral_network:
        :param upscale_network: Fully-connected network applied to input *before* FNO blocks
        :param downscale_network: Fully-connected network applied to input *after* FNO blocks
        :param padding: 0 for periodic domains. Positive for non-periodic domains
        :param kwargs: Keyword arguments for parent ``NeuralNetwork`` class
        """
        super().__init__(**kwargs)
        self.spectral_network = spectral_network
        self.upscale_network = upscale_network or nn.Linear(
            self.n_dimension, self.width  # todo: how do I get the dimension and width from spectral_network
        )
        self.downscale_network = downscale_network or nn.Linear(
            self.width, self.n_dimension
        )
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Input tensor
        :return: Output tensor
        """
        x = self.upscale_network(x)
        x = x.permute(0, 2, 1)
        if self.padding > 0:  # pad the domain if input is non-periodic
            x = F.pad(x, [0, self.padding])
        x = self.spectral_network(x)
        if self.padding > 0:  # pad the domain if input is non-periodic
            x = x[..., : -self.padding]
        x = x.permute(0, 2, 1)
        x = self.downscale_network(x)
        return x

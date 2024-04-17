import torch
import torch.nn as nn
import logging
from UQpy.scientific_machine_learning.baseclass.NeuralNetwork import NeuralNetwork


class VanillaNeuralNetwork(NeuralNetwork):

    def __init__(self, network: nn.Module, **kwargs):
        """Initialize a typical feed-forward neural network using the architecture provided by ``network``

        :param network: Network defining the function from :math:`f(x)=y`
        """
        super().__init__(**kwargs)
        self.network: nn.Module = network
        """Neural network architecture defined as a ``torch.nn.Module``"""

        self.logger = logging.getLogger(__name__)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward call of the neural network

        :param x: Input tensor
        :return: Output tensor
        """
        return self.network(x)

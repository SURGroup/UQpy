import torch.nn as nn
import logging
from UQpy.scientific_machine_learning.baseclass import NeuralNetwork


class FeedForwardNeuralNetwork(NeuralNetwork):

    def __init__(self, network: nn.Module):
        """Initialize a typical feed-forward neural network using the architecture provided by ``network``

        :param network: Network defining the function from :math:`f(x)=y`
        """
        super().__init__()
        self.network: nn.Module = network
        """Neural network architecture defined as a :py:class:`torch.nn.Module`"""

        # set all layers to the same training, dropping, and sampling mode
        training = False
        dropping = False
        sampling = False
        for m in self.network.modules():
            training = training or m.training
            if hasattr(m, "dropping"):
                dropping = dropping or m.dropping
            if hasattr(m, "sampling"):
                sampling = sampling or m.sampling
        self.train(training)
        self.drop(dropping)
        self.sample(sampling)

        self.logger = logging.getLogger(__name__)

    def forward(self, *args, **kwargs):
        """Forward call of the neural network

        :param args: Input arguments pass to ``network``
        :param kwargs: Keyword arguments passed to ``network``
        :return: Output arguments
        """
        return self.network(*args, **kwargs)

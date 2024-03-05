import torch.nn as nn
from abc import ABC, abstractmethod


class NeuralNetwork(ABC, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history = {"loss": None, "train loss": None, "test loss": None}
        self.is_deterministic = True

    @property
    @abstractmethod
    def _algorithm(self):
        """Implementation of `Pytorch optimization method`_

        .. _Pytorch optimization method: https://pytorch.org/docs/stable/optim.html
        """
        ...

    @property
    @abstractmethod
    def _loss_function(self):
        """Implementation of `Pytorch loss function`_

        .. _Pytorch loss function: https://pytorch.org/docs/stable/nn.html#loss-functions
        """
        ...

    @abstractmethod
    def forward(self):
        """Define the computation at every model call. Inherited from :code:`torch.nn.Module`.
        See `Pytorch documentation`_ for details

        .. _Pytorch documentation: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward
        """
        ...

    @abstractmethod
    def optimize(self):
        """Optimize network parameters using the error measured by :code:`loss_function`
        and algorithm defined by :code:`optimizer`.
        """
        ...

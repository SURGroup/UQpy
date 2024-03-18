import torch.nn as nn
from abc import ABC, abstractmethod


class NeuralNetwork(ABC, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history: dict = {"loss": None, "train loss": None, "test loss": None}
        """History of training and test loss during ``train`` method."""
        self.is_deterministic: bool = True

    @property
    @abstractmethod
    def optimizer(self):
        """Implementation of `Pytorch optimization method`_

        .. _Pytorch optimization method: https://pytorch.org/docs/stable/optim.html
        """
        ...

    @optimizer.setter
    @abstractmethod
    def optimizer(self, value):
        self._optimizer = value

    @property
    @abstractmethod
    def loss_function(self):
        """Implementation of `Pytorch loss function`_

        .. _Pytorch loss function: https://pytorch.org/docs/stable/nn.html#loss-functions
        """
        ...

    @loss_function.setter
    @abstractmethod
    def loss_function(self, value):
        self._loss_function = value

    @abstractmethod
    def forward(self, **kwargs):
        """Define the computation at every model call. Inherited from :code:`torch.nn.Module`.
        See `Pytorch documentation`_ for details

        .. _Pytorch documentation: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward
        """
        ...

    @abstractmethod
    def train(self, **kwargs):
        """Optimize network parameters using the error measured by :code:`loss_function`
        and algorithm defined by :code:`optimizer`.
        """
        ...

import torch.nn as nn
import torchinfo
from abc import ABC, abstractmethod


class NeuralNetwork(ABC, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def forward(self, **kwargs):
        """Define the computation at every model call. Inherited from :code:`torch.nn.Module`.
        See `Pytorch documentation`_ for details

        .. _Pytorch documentation: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward
        """
        ...

    def summary(self, **kwargs):
        """Call `torchinfo.summary()` on `self`"""
        return torchinfo.summary(self, **kwargs)

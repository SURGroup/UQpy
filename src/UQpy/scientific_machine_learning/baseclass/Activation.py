import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Activation(nn.Module, ABC):
    """This is an abstract baseclass for future development of Activation functions.
    As of September 2024, it is not used in UQpy.scientific_machine_learning
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def extra_repr(self) -> str:
        ...

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Activation(nn.Module, ABC):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def extra_repr(self) -> str:
        ...

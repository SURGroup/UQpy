import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class ActivationFunction(ABC, nn.Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def extra_repr(self) -> str:
        ...

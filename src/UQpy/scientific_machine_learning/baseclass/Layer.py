import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Layer(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset_parameters(self, a, b):
        """Fill all parameters with samples from :math:`\mathcal{U}(a, b)`"""
        for p in self.parameters():
            p.uniform_(a, b)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def extra_repr(self) -> str: ...

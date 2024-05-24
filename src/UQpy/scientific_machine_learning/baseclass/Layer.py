import torch.nn as nn
from abc import ABC, abstractmethod


class Layer(ABC, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def forward(self):
        ...

    @abstractmethod
    def extra_repr(self) -> str:
        ...

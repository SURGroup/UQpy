import torch.nn as nn
from abc import ABC, abstractmethod


class Layer(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def forward(self):
        ...

    @abstractmethod
    def extra_repr(self) -> str:
        ...

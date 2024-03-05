import torch.nn as nn
from abc import ABC, abstractmethod


class Loss(ABC, nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def forward(self):
        ...

import torch
from abc import ABC


class Optimizer(ABC, torch.optim.Optimizer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

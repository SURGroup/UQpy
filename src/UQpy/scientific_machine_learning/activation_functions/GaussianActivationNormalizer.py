import torch
from UQpy.scientific_machine_learning.baseclass import ActivationFunction
from beartype import beartype
from typing import Union
from UQpy.utilities.ValidationTypes import PositiveFloat


@beartype
class GaussianActivationNormalizer(ActivationFunction):
    def __init__(self, epsilon: PositiveFloat = 1e-8):
        """

        :param epsilon:
        """
        self.epsilon = epsilon
        self.mean: float = None
        self.std: float = None

    def forward(self, x: torch.Tensor, restore=False) -> torch.Tensor:
        """

        :param x:
        :param restore:
        :return:
        """
        if restore:
            if (self.mean is None) or (self.std is None):
                raise RuntimeError(
                    "UQpy: GaussianNormalizer called to restore but `mean` or `std` is None"
                )
            return (x * (self.std + self.epsilon)) + self.mean

        mean = torch.mean(x)
        std = torch.std(x)
        if self.mean is None:
            self.mean = mean
        if self.std is None:
            self.std = std
        return (x - self.mean) / (self.std + self.epsilon)

    def extra_repr(self) -> str:
        return "" if self.epsilon == 1e-8 else f"epsilon={self.epsilon}"

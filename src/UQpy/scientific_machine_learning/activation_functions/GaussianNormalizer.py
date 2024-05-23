import torch
from beartype import beartype
from UQpy.utilities.ValidationTypes import PositiveFloat


@beartype
class GaussianNormalizer:
    def __init__(self, x: torch.Tensor, epsilon: PositiveFloat = 1e-5, **kwargs):
        """Scale the tensor ``x`` to follow a standard normal distribution

        :param x: Input tensor
        :param epsilon: Small number to add to standard deviation for numerical stability
        :param kwargs: Keyword arguments for ``torch.mean`` and ``torch.sum``
        """
        self.x = x
        self.epsilon = epsilon

        self.mean = torch.mean(x, **kwargs)
        self.standard_deviation = torch.std(x, **kwargs)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Scale and shift the tensor ``x`` as ``y = (x - mean) / (std + epsilon)``

        :param x: Input tensor
        :return: Normalized tensor
        """
        return (x - self.mean) / (self.standard_deviation + self.epsilon)

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Restore the tensor ``y`` as ``x = (y * (std + epsilon)) + mean``

        :param y: Normalized tensor
        :return: Restore tensor
        """
        return (y * (self.standard_deviation + self.epsilon)) + self.mean

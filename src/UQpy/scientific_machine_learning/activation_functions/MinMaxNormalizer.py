import torch
from beartype import beartype


@beartype
class MinMaxNormalizer:
    # ToDo: isn't this the same as the range normalizer with low=0, high=1?
    def __init__(self, x: torch.Tensor):
        """Normalize the tensor x using min/max normalization

        :param x: Input tensor
        """
        self.x = x

        self.shift = torch.min(self.x)
        self.scale = torch.max(self.x) - torch.min(self.x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Input tensor
        :return: Normalized tensor
        """
        return (x - self.shift) / self.scale

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """

        :param y: Normalized tensor
        :return: Restore tensor
        """
        return (y * self.scale) + self.shift

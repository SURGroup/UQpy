import torch
from beartype import beartype


@beartype
class RangeNormalizer:
    def __init__(self, x: torch.Tensor, low: float = 0.0, high: float = 1.0):
        """Normalize ``x`` to fall within the range [``low``, ``high``]


        :param x: Input tensor
        :param low: Lower bound of the normalized range
        :param high: Upper bound of the normalized range
        """
        if low > high:
            raise ValueError("UQpy: `high` must be strictly greater than `low`")
        self.x = x
        self.low = low
        self.high = high

        x_min = torch.min(x)
        x_max = torch.max(x)
        self.scale = (self.high - self.low) / (x_max - x_min)
        self.shift = (-self.scale * x_max) + self.high

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Scale and shift ``x`` to fall within [low, high] as ``y = scale * x + shift``

        :param x: Input tensor
        :return: Normalized tensor
        """
        return (self.scale * x) + self.shift

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Restore ``y`` to its original range as ``x = (y -shift) / scale``

        :param y: Normalized tensor
        :return: Restored tensor
        """
        return (y - self.shift) / self.scale

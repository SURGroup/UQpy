import torch
from beartype import beartype
from typing import Union


@beartype
class RangeNormalizer:
    def __init__(
        self,
        x: torch.Tensor,
        low: Union[int, float] = 0.0,
        high: Union[int, float] = 1.0,
    ):
        """Normalize ``x`` to fall within the range [``low``, ``high``].

        Note: Due to machine precision, actual values may be outside of range by less than 1e-10.

        :param x: Input tensor.
        :param low: Lower bound of the normalized range
        :param high: Upper bound of the normalized range

        :raises ValueError: If ``low`` > `high`
        :raises RuntimeError: If x_min equals x_max
        """
        if low > high:
            raise ValueError("UQpy: `high` must be strictly greater than `low`")
        self.x = x
        self.low = low
        self.high = high

        x_min = torch.min(x)
        x_max = torch.max(x)
        if x_min == x_max:
            raise RuntimeError(
                "UQpy: RangeNormalizer is not defined if x_min is equal to x_max."
            )
        self.scale = (self.high - self.low) / (x_max - x_min)
        self.shift = self.low - (self.scale * x_min)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Scale and shift ``x`` to fall within [low, high] as ``y = (x * scale) + shift``

        :param x: Input tensor
        :return: Normalized tensor
        """
        return (x * self.scale) + self.shift

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Restore ``y`` to its original range as ``x = (y - shift) / scale``

        :param y: Normalized tensor
        :return: Restored tensor
        """
        return (y - self.shift) / self.scale

    def extra_repr(self) -> str:
        return f"low={self.low}, high={self.high}"

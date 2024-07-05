import torch
from UQpy.scientific_machine_learning.baseclass import Layer
from beartype import beartype
from typing import Union


@beartype
class RangeNormalizer(Layer):
    def __init__(
        self,
        x: torch.Tensor,
        low: Union[int, float] = 0.0,
        high: Union[int, float] = 1.0,
        encoding: bool = True,
        **kwargs,
    ):
        r"""Normalize a tensor to fall within the range :math:`[\text{low}, \text{high}]`

        Note: Due to machine precision, normalized values may be outside of range by errors on the order of :math:`10^{-8}`.

        :param x: Tensor of any shape
        :param low: Lower bound of the normalized range
        :param high: Upper bound of the normalized range
        :param encoding: If ``True``, scale and shift a tensor to be within :math:`[\text{low}, \text{high}]`.
         If ``False``, scale and shift from :math:`[\text{low}, \text{high}]` to the original range of ``x``.
         Default: ``True``


        :raises ValueError: If ``low`` greater or equal to ``high``
        :raises RuntimeError: If ``torch.min(x)`` equals ``torch.max(x)``

        Shape:

        - Input: Any shape
        - Output: Any shape (same shape as input)

        Example:

        >>> # use one instance and change mode
        >>> torch.manual_seed(0)  # for reproducibility
        >>> x = torch.normal(0, 1, (100, 100))
        >>> normalizer = sml.RangeNormalizer(x)
        >>> y = normalizer(x)
        >>> normalizer.decode()  # equivalent to normalizer.encode(False)
        >>> x_reconstruction = normalizer(y)
        >>> print(x.min(), y.min(), x_reconstruction.min())
        >>> print(x.max(), y.max(), x_reconstruction.max())
        tensor(-4.3433) tensor(0.) tensor(-4.3433)
        tensor(4.1015) tensor(1.) tensor(4.1015)

        >>> # use two instances with different modes
        >>> torch.manual_seed(0)  # for reproducibility
        >>> x = torch.normal(0, 1, (100, 100))
        >>> encoder = sml.RangeNormalizer(x)
        >>> decoder = sml.RangeNormalizer(x, encoding=False)
        >>> y = encoder(x)
        >>> x_reconstruction = decoder(y)
        >>> print(x.min(), y.min(), x_reconstruction.min())
        >>> print(x.max(), y.max(), x_reconstruction.max())
        tensor(-4.3433) tensor(0.) tensor(-4.3433)
        tensor(4.1015) tensor(1.) tensor(4.1015)

        """
        super().__init__(**kwargs)
        if low >= high:
            raise ValueError("UQpy: `high` must be strictly greater than `low`")
        self.x = x
        self.low = low
        self.high = high
        self.encoding = encoding

        x_min = torch.min(x)
        x_max = torch.max(x)
        if x_min == x_max:
            raise RuntimeError(
                "UQpy: RangeNormalizer is not defined if `torch.min(x)` is equal to `torch.max(x)`."
            )
        self.scale = (self.high - self.low) / (x_max - x_min)
        """Multiplicative factor to rescale range of x to interval width"""
        self.shift = self.low - (self.scale * x_min)
        """Additive factor to make interval start at ``low``"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale and shift ``x`` to fall within ``[self.low, self.high]``

        :param x: Tensor of any shape
        :return: Tensor of same shape as ``x``
        """
        if self.encoding:
            return self.scale_down(x)
        else:
            return self.scale_up(x)

    def scale_down(self, x: torch.Tensor) -> torch.Tensor:
        """Scale and shift ``x`` as ``(x * scale) + shift``

        :param x: Tensor of any shape
        :return: Normalized tensor of same shape as ``x``
        """
        return (x * self.scale) + self.shift

    def scale_up(self, y: torch.Tensor) -> torch.Tensor:
        """Restore the tensor ``y`` as ``x = (y - shift) / scale``

        :param y: Normalized tensor of any shape
        :return: Restored tensor of same shape as ``x``
        """
        return (y - self.shift) / self.scale

    def encode(self, mode: bool = True):
        """Set the normalizer to scale and shift a tensor to fall within range :math:[\text

        :param mode: If ``True``, set ``self.encoding`` to ``True``. Default: ``True``
        """
        self.encoding = mode

    def decode(self, mode: bool = True):
        """Set the normalizer to restore a tensor to its original range

        :param mode: If ``True``, set ``self.encoding`` to ``False``. Default: ``True``
        """
        self.encoding = not mode

    def extra_repr(self) -> str:
        return f"low={self.low}, high={self.high}, encoding={self.encoding}"

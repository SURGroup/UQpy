import torch
from UQpy.scientific_machine_learning.baseclass import Layer
from UQpy.utilities.ValidationTypes import NonNegativeInteger
from beartype import beartype
from beartype.vale import Is
from typing import Union, Annotated


@beartype
class RangeNormalizer(Layer):
    def __init__(
        self,
        x: torch.Tensor,
        encoding: bool = True,
        low: Union[int, float] = 0.0,
        high: Union[int, float] = 1.0,
        dim: Union[
            NonNegativeInteger,
            Annotated[tuple, Is[lambda x: all([isinstance(d, int) for d in x])]],
            None,
        ] = None,
    ):
        r"""Normalize a tensor to fall within the range :math:`[\text{low}, \text{high}]`

        .. note:: Due to machine precision, normalized values may be outside of range by errors on the order of :math:`10^{-8}`.

        :param x: Tensor of any shape
        :param encoding: If ``True``, scale and shift a tensor to be within :math:`[\text{low}, \text{high}]`.
         If ``False``, scale and shift from :math:`[\text{low}, \text{high}]` to the original range of ``x``.
         Default: ``True``
        :param low: Lower bound of the normalized range
        :param high: Upper bound of the normalized range
        :param dim: Dimensions to be reduced in :math:`\min(x), \max(x)`.
         If :code:`None`, reduce all dimensions for scalar min and max. Default: None

        :raises ValueError: If ``low`` greater or equal to ``high``
        :raises RuntimeError: If :math:`\min(x)` equals :math:`\max(x)` over any dimension to be reduced.
         This is to prevent a :code:`ZeroDivisionError` from occuring in the computation of :code:`scale`

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
        super().__init__()
        if low >= high:
            raise ValueError(
                f"UQpy: Invalid combination low={low} and high={high}. "
                f"`high` must be strictly greater than `low`"
            )
        self.x = x
        self.low = low
        self.high = high
        self.encoding = encoding
        self.dim = dim

        # handle cases if dim is None, an integer, or a tuple
        if self.dim is None:
            x_min = torch.min(x)
            x_max = torch.max(x)
        elif isinstance(self.dim, int):
            x_min, _ = torch.min(x, dim=self.dim, keepdim=True)
            x_max, _ = torch.max(x, dim=self.dim, keepdim=True)
        elif isinstance(self.dim, tuple):
            x_min = x
            x_max = x
            for d in self.dim:
                x_min, _ = torch.min(x_min, dim=d, keepdim=True)
                x_max, _ = torch.max(x_max, dim=d, keepdim=True)
        else:  # else statement is redundant due to beartype hinting, but better to be safe than sorry
            raise ValueError(
                f"UQpy: Invalid dim={dim}. Must be one of None, int, or tuple of ints."
            )

        if torch.any(x_min == x_max):  # if x_min equals x_max, a divide by zero error will occur when computing scale
            raise RuntimeError(
                "UQpy: RangeNormalizer is not defined if min(x) is equal to max(x) over any dimension to be reduced."
            )
        if torch.any(torch.isnan(x_min)) or torch.any(torch.isinf(x_min)):
            raise RuntimeError(
                "UQpy: Invalid value for min(x) in dimension to be reduced. The min cannot be `nan` or `inf`."
            )
        if torch.any(torch.isnan(x_max)) or torch.any(torch.isinf(x_max)):
            raise RuntimeError(
                "UQpy: Invalid value for max(x) in dimension to be reduced. The max cannot be `nan` or `inf`."
            )

        self.scale: torch.Tensor = (self.high - self.low) / (x_max - x_min)
        """Multiplicative factor to rescale range of x to interval width"""
        self.shift: torch.Tensor = self.low - (self.scale * x_min)
        """Additive factor to make interval start at ``self.low``"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Scale and shift ``x`` to fall within a new range.

        If :code:`self.encoding` is :code:`True`, return :math:`(x \times \text{scale}) + \text{shift}`.
        If :code:`self.encoding` is :code:`False`, return :math:`\frac{x - \text{shift}}{\text{scale}}`

        :param x: Tensor of any shape
        :return: Tensor of same shape as ``x``
        """
        if self.encoding:
            return (x * self.scale) + self.shift
        else:
            return (x - self.shift) / self.scale

    def encode(self, mode: bool = True):
        """Set the normalizer to scale and shift a tensor to fall within range :math:`[\text{low}, \text{high}]`

        :param mode: If ``True``, set ``self.encoding`` to ``True``. Default: ``True``
        """
        self.encoding = mode

    def decode(self, mode: bool = True):
        """Set the normalizer to restore a tensor to its original range

        :param mode: If ``True``, set ``self.encoding`` to ``False``. Default: ``True``
        """
        self.encoding = not mode

    def extra_repr(self) -> str:
        s = "encoding={encoding}"
        if self.low != 0.0:
            s += ", low={low}"
        if self.high != 1.0:
            s += ", high={high}"
        if self.dim is not None:
            s += ", dim={dim}"
        return s.format(**self.__dict__)

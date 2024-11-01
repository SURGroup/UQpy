import torch
from UQpy.scientific_machine_learning.baseclass import Layer
from UQpy.utilities.ValidationTypes import PositiveFloat, NonNegativeInteger
from typing import Union, Annotated
from beartype import beartype
from beartype.vale import Is


@beartype
class GaussianNormalizer(Layer):
    def __init__(
        self,
        x: torch.Tensor,
        encoding: bool = True,
        epsilon: PositiveFloat = 1e-8,
        dim: Union[
            NonNegativeInteger,
            Annotated[tuple, Is[lambda x: all([isinstance(d, int) for d in x])]],
            None,
        ] = None,
    ):
        r"""Normalize a tensor to have mean of zero and standard deviation of one.

        .. note::
            Due to machine percision, mean and standard deviation may have errors on the order of :math:`10^{-8}`.
            Using different data types may affect percision of results.

        :param x: Tensor of any shape
        :param encoding: If ``True``, scale and shift a tensor to have mean of zero and standard deviation of one.
         If ``False``, scale and shift from a mean of zero and std of one to the original mean and std of ``x``.
         Default: ``True``
        :param epsilon: Small positive value added to standard deviation for numerical stability
        :param dim: Dimensions to be reduced in :math:`\text{mean}(x), \text{std}(x)`.
         If :code:`None`, reduce all dimensions for scalar min and max. Default: None

        :raises RuntimeError: If ``torch.mean(x)`` or ``torch.std(x)`` contains infinite (:code:`inf`)
         or not-a-number (:code:`nan`) over a dimension to be reduced

        Shape:

        - Input: Any shape
        - Output: Any shape (same shape as input)

        Example:

        >>> # use one instance and change mode
        >>> torch.manual_seed(0)  # for reproducibility
        >>> x = torch.rand(100, 100)
        >>> normalizer = sml.GaussianNormalizer(x)
        >>> y = normalizer(x)
        >>> normalizer.decode()  # equivalent to normalizer.encode(False)
        >>> x_reconstruction = normalizer(y)
        >>> print(x.mean(), y.mean(), x_reconstruction.mean())
        >>> print(x.std(), y.std(), x_reconstruction.std())
        tensor(0.5003) tensor(-6.5804e-09) tensor(0.5003)
        tensor(0.2879) tensor(1.) tensor(0.2879)

        >>> # use two instances with different modes
        >>> torch.manual_seed(0)  # for reproducibility
        >>> x = torch.rand(100, 100)
        >>> encoder = sml.GaussianNormalizer(x)
        >>> decoder = sml.GaussianNormalizer(x, encoding=False)
        >>> y = encoder(x)
        >>> x_reconstruction = decoder(y)
        >>> print(x.mean(), y.mean(), x_reconstruction.mean())
        >>> print(x.std(), y.std(), x_reconstruction.std())
        tensor(0.5003) tensor(-6.5804e-09) tensor(0.5003)
        tensor(0.2879) tensor(1.) tensor(0.2879)
        """
        super().__init__()
        self.x = x
        self.epsilon = epsilon
        self.encoding = encoding
        self.dim = dim

        self.mean: torch.Tensor = torch.mean(self.x, dim=self.dim, keepdim=True)
        """Original mean of the initialization tensor"""
        self.std: torch.Tensor = torch.std(self.x, dim=self.dim, keepdim=True)
        """Original standard deviation of the initialization tensor"""

        if torch.any(torch.isnan(self.mean)) or torch.any(torch.isinf(self.mean)):
            raise RuntimeError(
                "UQpy: Invalid value for mean(x) in dimension to be reduced. The mean cannot be `nan` or `inf`."
            )
        if torch.any(torch.isnan(self.std)) or torch.any(torch.isinf(self.std)):
            raise RuntimeError(
                "UQpy: Invalid value for std(x) in dimension to be reduced. The std cannot be `nan` or `inf`."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Scale ``x`` to have a new mean and standard deviation.

        If :code:`self.encoding` is :code:`True`, return :math:`\frac{x - \text{mean}}{\text{std} + \epsilon}`.
        If :code:`self.encoding` is :code:`False`, return :math:`(x \times (\text{std} + \epsilon)) + \text{mean}`

        :param x: Tensor of any shape
        :return: Tensor of same shape as ``x``
        """
        if self.encoding:
            return (x - self.mean) / (self.std + self.epsilon)
        else:
            return (x * (self.std + self.epsilon)) + self.mean

    def encode(self, mode: bool = True):
        """Set the Normalizer to scale a tensor a mean of zero and standard deviation of one

        :param mode: If :code:`True`, set ``self.encoding`` to :code:`True`. Default: :code:`True`
        """
        self.encoding = mode

    def decode(self, mode: bool = True):
        """Set the normalizer to restore a tensor to its original mean and standard deviation

        :param mode: If :code:`True`, set ``self.encoding`` to :code:`False`. Default: :code:`True`
        """
        self.encoding = not mode

    def extra_repr(self) -> str:
        s = "encoding={encoding}"
        if self.epsilon != 1e-8:
            s += ", epsilon={epsilon}"
        if self.dim is not None:
            s += ", dim={dim}"
        return s.format(**self.__dict__)

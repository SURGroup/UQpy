import torch
from UQpy.scientific_machine_learning.baseclass import Layer
from beartype import beartype
from UQpy.utilities.ValidationTypes import PositiveFloat


@beartype
class GaussianNormalizer(Layer):
    def __init__(
        self,
        x: torch.Tensor,
        epsilon: PositiveFloat = 1e-8,
        encoding: bool = True,
        **kwargs,
    ):
        """Normalize a tensor to have mean of zero and standard deviation of one.

        Note: Due to machine percision, mean and standard deviation may have errors on the order of :math:`10^{-8}`.

        :param x: Tensor of any shape
        :param epsilon: Small positive value added to standard deviation for numerical stability
        :param encoding: If ``True``, scale and shift a tensor to have mean of zero and std of one.
         If ``False``, scale and shift from a mean of zero and std of one to the original mean and std of ``x``.
         Default: ``True``

         :raises RuntimeError: If ``torch.mean(x)`` or ``torch.std(x)`` is infinite (``inf``) or not-a-number (``nan``)

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
        super().__init__(**kwargs)
        self.x = x
        self.epsilon = epsilon
        self.encoding = encoding

        self.mean: torch.Tensor = torch.mean(self.x)
        """Original mean of the initialization tensor"""
        self.std: torch.Tensor = torch.std(self.x)
        """Original standard deviation of the initialization tensor"""

        if torch.isinf(self.mean):
            raise RuntimeError(
                "UQpy: Invalid value for `torch.mean(x)`. The mean cannot be `inf`."
            )
        if torch.isnan(self.mean):
            raise RuntimeError(
                "UQpy: Invalid value for `torch.mean(x)`. The mean cannot be `nan`."
            )
        if torch.isinf(self.std):
            raise RuntimeError(
                "UQpy: Invalid value for `torch.std(x)`. The std cannot be `inf`."
            )
        if torch.isnan(self.std):
            raise RuntimeError(
                "UQpy: Invalid value for `torch.std(x)`. The std cannot be `nan`."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale ``x`` to have a new mean and standard deviation

        :param x: Tensor of any shape
        :return: Tensor of same shape as ``x``
        """
        if self.encoding:
            return self.scale_down(x)
        else:
            return self.scale_up(x)

    def scale_down(self, x: torch.Tensor) -> torch.Tensor:
        """Scale and shift the tensor ``x`` as ``y = (x - mean) / (std + epsilon)``

        :param x: Tensor of any shape
        :return: Normalized tensor of same shape as ``x``
        """
        return (x - self.mean) / (self.std + self.epsilon)

    def scale_up(self, y: torch.Tensor) -> torch.Tensor:
        """Restore the tensor ``y`` as ``x = (y * (std + epsilon)) + mean``

        :param y: Normalized tensor of any shape
        :return: Restored tensor of same shape as ``x``
        """
        return (y * (self.std + self.epsilon)) + self.mean

    def encode(self, mode: bool = True):
        """Set the Normalizer to scale a tensor a mean of zero and standard deviation of one

        :param mode: If ``True``, set ``self.encoding`` to ``True``. Default: ``True``
        """
        self.encoding = mode

    def decode(self, mode: bool = True):
        """Set the normalizer to restore a tensor to its original mean and standard deviation

        :param mode: If ``True``, set ``self.encoding`` to ``False``. Default: ``True``
        """
        self.encoding = not mode

    def extra_repr(self) -> str:
        return f"epsilon={self.epsilon}, encoding={self.encoding}"

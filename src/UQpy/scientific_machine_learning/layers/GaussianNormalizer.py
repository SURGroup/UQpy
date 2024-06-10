import torch
from UQpy.scientific_machine_learning.baseclass import Layer
from beartype import beartype
from UQpy.utilities.ValidationTypes import PositiveFloat


@beartype
class GaussianNormalizer(Layer):
    def __init__(self, x: torch.Tensor, epsilon: PositiveFloat = 1e-8, encoding: bool = True, **kwargs):
        """

        :param x:
        :param epsilon:
        :param kwargs:

        Shape:

        - Input: Any shape
        - Output: Any shape (same shape as input)

        Example:

        >>> x = torch.rand(100, 100)
        >>> encoder = GaussianNormalizer(x)
        >>> decoder = GaussianNormalizer(x, encoding=False)
        >>> y = encoder(x)
        >>> x_reconstruction = decoder(y)
        >>> print(x.mean(), y.mean(), x_reconstruction.mean())
        >>> print(x.std(), y.std(), x_reconstruction.std())
        """
        super().__init__(**kwargs)
        self.x = x
        self.epsilon = epsilon
        self.encoding = encoding

        self.mean = torch.mean(self.x)
        self.std = torch.std(self.x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
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
        """

        :param mode:
        :return:
        """
        self.encoding = mode

    def extra_repr(self) -> str:
        return f"epsilon={self.epsilon}, encoding={self.encoding}"

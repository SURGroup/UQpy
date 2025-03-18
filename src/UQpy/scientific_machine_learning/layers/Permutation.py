import torch
from UQpy.scientific_machine_learning.baseclass import Layer


class Permutation(Layer):
    def __init__(self, dims: tuple[int], **kwargs):
        """Permute the dimensions of a tensor.

        See :py:class:`torch.permute` for documentation

        :param dims: Dimensions passed to :code:`torch.permute`
        """
        super().__init__(**kwargs)
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calls ``torch.permute(x, dims)``

        :param x: Tensor of any shape
        :return: Tensor of permuted shape
        """
        return torch.permute(x, self.dims)

    def extra_repr(self) -> str:
        return f"dims={self.dims}"

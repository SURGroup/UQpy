import torch
import torch.nn as nn


class Permutation(nn.Module):
    def __init__(self, dims: tuple[int], **kwargs):
        """Permute the dimensions of a tensor. See ``torch.permute`` for documentation

        :param dims: Dimensions passed to ``torch.permute``
        """
        super().__init__(**kwargs)
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, *self.dims)

    def extra_repr(self) -> str:
        return f"dims={self.dims}"

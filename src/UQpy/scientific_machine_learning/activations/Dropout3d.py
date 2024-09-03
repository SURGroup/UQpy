import torch
import torch.nn.functional as F
from UQpy.scientific_machine_learning.baseclass import DropoutActivation
from typing import Annotated
from beartype import beartype
from beartype.vale import Is


@beartype
class Dropout3d(DropoutActivation):
    def __init__(
        self,
        p: Annotated[float, Is[lambda p: 0 <= p <= 1]] = 0.5,
        inplace: bool = False,
        dropping: bool = True,
        **kwargs
    ):
        """Randomly zero out entire channels with probability :math:`p`

        A channel is a 3D feature map.

        :param p: Probability of a channel to be zeroed. Default: 0.5
        :param inplace: If ``True``, will do this operation in-place. Default: ``False``
        :param dropping: If ``True``, will perform dropout, otherwise acts as identity function. Default: ``True``

        Shape:

        - Input: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`
        - Output: :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)` (same shape as input)

        Example:

        >>> dropout = sml.Dropout3d(p=0.4)
        >>> input = torch.rand(2, 4, 8, 16, 32)
        >>> output = dropout(input)
        """
        super().__init__(**kwargs)
        self.p = p
        self.inplace = inplace
        self.dropping = dropping

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calls :func:`torch.nn.functional.dropout3d`

        :param x: Tensor of shape :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)`
        :return: Tensor of shape :math:`(N, C, D, H, W)` or :math:`(C, D, H, W)` (same shape as ``x``)
        """
        return F.dropout3d(x, self.p, self.dropping, self.inplace)

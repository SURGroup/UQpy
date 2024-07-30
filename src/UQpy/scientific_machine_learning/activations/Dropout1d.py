import torch
import torch.nn.functional as F
from UQpy.scientific_machine_learning.baseclass import DropoutActivation
from typing import Annotated
from beartype import beartype
from beartype.vale import Is


@beartype
class Dropout1d(DropoutActivation):
    def __init__(
        self,
        p: Annotated[float, Is[lambda p: 0 <= p <= 1]] = 0.5,
        inplace: bool = False,
        dropping: bool = True,
        **kwargs
    ):
        """Randomly zero out entire channels with probability :math:`p`

        A channel is a 1D feature map

        :param p: Probability of a channel to be zeroed. Default: 0.5
        :param inplace: If ``True``, will do this operation in-place. Default: ``False``
        :param dropping: If ``True``, will perform dropout, otherwise acts as identity function. Default: ``True``

        Shape:

        - Input: :math:`(N, C, L)` or :math:`(C, L)`
        - Output: :math:`(N, C, L)` or :math:`(C, L)`  (same shape as input)

        Example:

        >>> dropout = sml.Dropout1d(p=0.6)
        >>> input = torch.rand(10, 3, 200)
        >>> output = dropout(input)
        """
        super().__init__(**kwargs)
        self.p = p
        self.inplace = inplace
        self.dropping = dropping

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calls ``torch.nn.functional.dropout1d`` on ``x``

        :param x: Tensor of shape :math:`(N, C, L)` or :math:`(C, L)`
        :return: Tensor of same shape as ``x``
        """
        return F.dropout1d(x, self.p, self.dropping, self.inplace)

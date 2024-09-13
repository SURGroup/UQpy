import torch
import torch.nn.functional as F
from UQpy.scientific_machine_learning.baseclass import ProbabilisticDropoutLayer
from typing import Annotated
from beartype import beartype
from beartype.vale import Is


@beartype
class ProbabilisticDropout(ProbabilisticDropoutLayer):
    def __init__(
        self,
        p: Annotated[float, Is[lambda p: 0 <= p <= 1]] = 0.5,
        inplace: bool = False,
        dropping: bool = True,
        **kwargs
    ):
        """Randomly zero out some elements of the input tensor with probability :math:`p`

        :param p: Probability of an element to be zeroed. Default: 0.5
        :param inplace: If ``True``, will do this operation in-place. Default: ``False``
        :param dropping: If ``True``, will perform dropout, otherwise acts as identity function. Default: ``True``

        Shape:

        - Input: Any shape
        - Output: Any shape (same shape as input)

        Example:

        >>> dropout = sml.ProbabilisticDropout(p=0.75)
        >>> input = torch.rand(12, 100)
        >>> output = dropout(input)
        """
        super().__init__(**kwargs)
        self.p = p
        self.inplace = inplace
        self.dropping = dropping

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Calls :func:`torch.nn.functional.dropout`

        :param x: Tensor of any shape
        :return: Tensor of same shape as ``x``
        """
        return F.dropout(x, self.p, self.dropping, self.inplace)

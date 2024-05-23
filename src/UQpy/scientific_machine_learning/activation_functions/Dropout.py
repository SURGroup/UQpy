import torch
import torch.nn.functional as F
from UQpy.scientific_machine_learning.baseclass import ActivationFunction
from beartype import beartype
from beartype.vale import Is
from typing import Annotated
from abc import ABC


@beartype
class _Dropout(ABC, ActivationFunction):
    def __init__(
        self,
        p: Annotated[float, Is[lambda p: 0 <= p <= 1]] = 0.5,
        inplace: bool = False,
        dropping: bool = True,
    ):
        """

        :param p: Probability of an element to be zeroed. Default: 0.5
        :param inplace: If set to ``True``, will do this operation in-place. Default: ``False``
        :param dropping: If set to ``True``, randomly zeros some tensor elements.
         If ``False``, acts as identity function. Default: ``True``
        """
        super().__init__()
        self.p = p
        self.inplace = inplace
        self.dropping = dropping

    def drop(self, mode: bool = True):
        """Set dropping mode.

        :param mode: If ``True``, layer parameters are dropped.
        """
        self.dropping = mode

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace={self.inplace}, dropping={self.dropping}"


class Dropout(_Dropout):
    """Randomly zero out elements."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, self.p, self.dropping, self.inplace)


class Dropout1d(_Dropout):
    """Randomly zero out entire 1D feature maps."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout1d(x, self.p, self.dropping, self.inplace)


class Dropout2d(_Dropout):
    """Randomly zero out entire 2D feature maps"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout2d(x, self.p, self.dropping, self.inplace)


class Dropout3d(_Dropout):
    """Randomly zero out entire 3D feature maps"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout3d(x, self.p, self.dropping, self.inplace)

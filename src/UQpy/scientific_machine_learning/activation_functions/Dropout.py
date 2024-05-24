import torch
import torch.nn.functional as F
from UQpy.scientific_machine_learning.baseclass import ActivationFunction
from beartype import beartype
from beartype.vale import Is
from typing import Annotated


@beartype
class _Dropout(ActivationFunction):
    def __init__(
        self,
        p: Annotated[float, Is[lambda p: 0 <= p <= 1]] = 0.5,
        inplace: bool = False,
        dropping: bool = True,
        **kwargs
    ):
        """Randomly zero out some elements of a tensor

        :param p: Probability of an element to be zeroed. Default: 0.5
        :param inplace: If set to ``True``, will do this operation in-place. Default: ``False``
        :param dropping: If set to ``True``, randomly zeros some tensor elements. Default: ``True``
        """
        super().__init__(**kwargs)
        self.p = p
        self.inplace = inplace
        self.dropping = dropping

    def drop(self, mode: bool = True):
        """Set dropping mode.

        :param mode: If ``True``, layer parameters are dropped.
        """
        self.dropping = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def extra_repr(self) -> str:
        keyword_args = []
        if self.p != 0.5:
            keyword_args.append("p={p}")
        if self.inplace:
            keyword_args.append("inplace={inplace}")
        if not self.dropping:
            keyword_args.append("dropping={dropping}")
        if not keyword_args:
            return ""
        else:
            s = ", ".join(keyword_args)
            return s.format(**self.__dict__)


class Dropout(_Dropout):
    """Randomly zero out elements."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(x, self.p, self.dropping, self.inplace)


class Dropout1d(_Dropout):
    """Randomly zero out entire 1D feature maps."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout1d(x, self.p, self.dropping, self.inplace)


class Dropout2d(_Dropout):
    """Randomly zero out entire 2D feature maps."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout2d(x, self.p, self.dropping, self.inplace)


class Dropout3d(_Dropout):
    """Randomly zero out entire 3D feature maps."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout3d(x, self.p, self.dropping, self.inplace)

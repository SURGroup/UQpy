import torch
from UQpy.scientific_machine_learning.baseclass.Layer import Layer
from abc import ABC, abstractmethod
from beartype import beartype
from beartype.vale import Is
from typing import Annotated


@beartype
class ProbabilisticDropoutLayer(Layer, ABC):
    def __init__(
        self,
        p: Annotated[float, Is[lambda p: 0 <= p <= 1]] = 0.5,
        inplace: bool = False,
        dropping: bool = True,
        **kwargs
    ):
        """Randomly zero out some elements of a tensor

        :param p: Probability of an element to be zeroed. Default: 0.5
        :param inplace: If ``True``, will do this operation in-place. Default: ``False``
        :param dropping: If ``True``, will perform dropout, otherwise acts as identity function. Default: ``True``
        """
        super(Layer, self).__init__()
        self.p = p
        self.inplace = inplace
        self.dropping = dropping

    def drop(self, mode: bool = True):
        """Set dropping mode.

        :param mode: If ``True``, will perform dropout, otherwise acts as identity function. Default: ``True``
        """
        self.dropping = mode

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def extra_repr(self) -> str:
        s = "p={p}, dropping={dropping}"
        if self.inplace:
            s += "inplace={inplace}"
        return s.format(**self.__dict__)

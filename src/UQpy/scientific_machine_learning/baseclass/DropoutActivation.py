from UQpy.scientific_machine_learning.baseclass import Activation
from beartype import beartype
from beartype.vale import Is
from typing import Annotated
from abc import ABC, abstractmethod


@beartype
class DropoutActivation(Activation, ABC):
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
        super().__init__(**kwargs)
        self.p = p
        self.inplace = inplace
        self.dropping = dropping

    def drop(self, mode: bool = True):
        """Set dropping mode.

        :param mode: If ``True``, will perform dropout, otherwise acts as identity function.
        """
        self.dropping = mode

    @abstractmethod
    def forward(self):
        ...

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

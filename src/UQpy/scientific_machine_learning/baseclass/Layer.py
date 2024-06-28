import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class Layer(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sampling: bool = False
        self.dropping: bool = False

    @abstractmethod
    def forward(self): ...

    @abstractmethod
    def extra_repr(self) -> str: ...

    def sample(self, mode: bool = True):
        """Set sampling mode for Neural Network and all child modules

        Note: This method and ``self.sampling`` only affects Bayesian layers

        :param mode: If ``True, sample from distributions, otherwise use distribution means.
        :return: ``self``
        """
        self.sampling = mode
        self.apply(self.__set_sampling)
        return self

    @torch.no_grad()
    def __set_sampling(self, m):
        if hasattr(m, "sampling"):
            m.sampling = self.sampling

    def drop(self, mode: bool = True):
        """Set dropping mode.

        Note: This method and ``self.dropping`` only affects dropout layers

        :param mode: If ``True``, will perform dropout, otherwise acts as identity function.
        """
        self.dropping = mode
        self.apply(self.__set_dropping)
        return self

    @torch.no_grad()
    def __set_dropping(self, m):
        if hasattr(m, "dropping"):
            m.dropping = self.dropping

import torch
import torch.nn as nn
import torchinfo
from abc import ABC, abstractmethod


class NeuralNetwork(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sampling: bool = True
        """Boolean represents whether this module is in sampling mode or not."""
        self.dropping: bool = True
        """Boolean represents whether this module is in dropping mode or not."""

    @abstractmethod
    def forward(self, **kwargs):
        """Define the computation at every model call. Inherited from :code:`torch.nn.Module`.
        See `Pytorch documentation`_ for details

        .. _Pytorch documentation: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward
        """
        ...

    def summary(self, **kwargs):
        """Call ``torchinfo.summary()`` on ``self``.
        See `torchinfo documentation <https://github.com/TylerYep/torchinfo?tab=readme-ov-file#documentation>`__
        for details

        :param kwargs: Keyword arguments passed to ``torchinfo.summary``.
        :return: Model statistics
        """
        return torchinfo.summary(self, **kwargs)

    def count_parameters(self):
        """Get the total number of parameters that require a gradient computation in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def sample(self, mode: bool = True):
        """Set sampling mode.

        Note: This method and ``self.sampling`` only affects UQpy's Bayesian layers

        :param mode: If ``True`` sample from distributions, otherwise use distribution means.
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

        Note: This method and ``self.dropping`` only affects UQpy's dropout layers

        :param mode: If ``True`` perform dropout, otherwise act as the identity function.
        """
        self.dropping = mode
        self.apply(self.__set_dropping)
        return self

    @torch.no_grad()
    def __set_dropping(self, m):
        if hasattr(m, "dropping"):
            m.dropping = self.dropping

    def is_deterministic(self) -> bool:
        """Check if neural network is behaving deterministically or probabilistically.

        Note: This flag may be incorrect if the model has sources of randomness that do not depend on
        ``training``, ``dropping``, or ``sampling``.

        :return: ``True`` if ``sampling``, ``dropping``, and ``training`` are all ``False``.
         Otherwise, returns ``False.
        """
        return not (self.sampling or self.dropping or self.training)

    def set_deterministic(self, mode: bool = True):
        """Set training, dropping, and sampling to the *opposite* of ``mode``.

        This is equivalent to

        >>> model.train(not mode)
        >>> model.drop(not mode)
        >>> model.sample(not mode)

        If the model has sources of randomness that do not depend on the ``training``, ``dropping``, or ``sampling``
        attributes, they will not be affected.
        """
        self.train(not mode)
        self.drop(not mode)
        self.sample(not mode)

import itertools
from abc import ABC, abstractmethod
import numpy as np


class Kernel(ABC):
    """
    This is a blueprint for kernels implemented in the :py:mod:`kernels` module .
    """

    @abstractmethod
    def kernel_entry(self, xi, xj):
        """
        Given two points this method calculates the respective kernel entry. Each concrete kernel implementation must
        override this method and provide its own implementation

        :param xi: First point.
        :param xj: Second point.
        :return: Float representing the kernel entry.
        """
        pass

import itertools
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from UQpy.utilities.ValidationTypes import NumpyFloatArray


class Kernel(ABC):
    """
    This is a blueprint for kernels implemented in the :py:mod:`kernels` module .
    """
    def __init__(self):
        self.kernel_matrix = None
        """Kernel matrix defining the similarity between the points"""

    def calculate_kernel_matrix(self, points: Union[list, NumpyFloatArray]):
        pass

    @abstractmethod
    def _kernel_entry(self, xi, xj):
        """
        Given two points this method calculates the respective kernel entry. Each concrete kernel implementation must
        override this method and provide its own implementation

        :param xi: First point.
        :param xj: Second point.
        :return: Float representing the kernel entry.
        """
        pass

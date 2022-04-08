import itertools
from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from UQpy.utilities import GrassmannPoint
from UQpy.utilities.ValidationTypes import NumpyFloatArray, Numpy2DFloatArray


class Kernel(ABC):
    """
    This is the baseclass for all kernels in :py:mod:`UQpy`.

    This serves as a blueprint to show the methods for kernels implemented in the :py:mod:`.kernels` module .
    """
    def __init__(self):
        self.kernel_matrix: np.ndarray = None
        """Kernel matrix defining the similarity between the points"""

    def calculate_kernel_matrix(self, points: Union[Numpy2DFloatArray, list[GrassmannPoint]]):
        """
        Using the kernel-specific :py:meth:`.kernel_entry` method, this function assembles the kernel matrix.

        :param points: Set of data points. Depending on the type of kernel these should be either :class:`numpy.ndarray`
            or of type :class:`.GrassmannPoint`.
        """
        pass

    def optimize_parameters(self, data: Union[Numpy2DFloatArray, list[GrassmannPoint]], **kwargs_optimization):
        """
        This serves as a blueprint function in case a kernel provides the ability to optimize its parameters. In that
        case, the kernel will override this method, and store the optimized parameters in the kernel's attributes.

        :param data: Set of data points.
        :param kwargs_optimization: Keyword arguments containing any extra parameters needed to perform the
            optimization. These will be unique to the specific kernel.
        """
        pass

    @abstractmethod
    def kernel_entry(self, xi: Union[Numpy2DFloatArray, GrassmannPoint],
                     xj: Union[Numpy2DFloatArray, GrassmannPoint]):
        """
        Given two points, this method calculates the respective kernel entry. Each concrete kernel implementation must
        override this method and provide its own implementation.

        :param xi: First point.
        :param xj: Second point.
        :return: Float representing the kernel entry.
        """
        pass

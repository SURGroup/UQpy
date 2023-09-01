import itertools
from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np

from UQpy.utilities.GrassmannPoint import GrassmannPoint
from UQpy.utilities.kernels.baseclass.Kernel import Kernel


class GrassmannianKernel(Kernel, ABC):
    """The parent class for Grassmannian kernels implemented in the :py:mod:`kernels` module ."""

    def __init__(self, kernel_parameter: Union[int, float] = None):
        """
        :param kernel_parameter: Number of independent p-planes of each Grassmann point.
        """
        super().__init__(kernel_parameter)

    def calculate_kernel_matrix(self, x: list[GrassmannPoint], s: list[GrassmannPoint]):
        p = self.kernel_parameter
        list1 = [point.data if not p else point.data[:, :p] for point in x]
        list2 = [point.data if not p else point.data[:, :p] for point in s]
        product = [self.element_wise_operation(point_pair)
                   for point_pair in list(itertools.product(list1, list2))]
        self.kernel_matrix = np.array(product).reshape(len(list1), len(list2))
        return self.kernel_matrix

    @abstractmethod
    def element_wise_operation(self, xi_j: Tuple) -> float:
        pass

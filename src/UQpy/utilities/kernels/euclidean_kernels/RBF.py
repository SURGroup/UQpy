from typing import Union

import numpy as np

from UQpy.utilities.kernels.baseclass.EuclideanKernel import EuclideanKernel
from UQpy.utilities.kernels.baseclass.Kernel import Kernel


class RBF(EuclideanKernel):
    def __init__(self, kernel_parameter: Union[int, float] = 1.0):
        super().__init__(kernel_parameter)

    def calculate_kernel_matrix(self, x, s):
        """
        This method compute the RBF kernel on sample points 'x' and 's'.

        :params x: An array containing training points.
        :params s: An array containing input points.
        """
        stack = Kernel.check_samples_and_return_stack(x / self.kernel_parameter, s / self.kernel_parameter)
        self.kernel_matrix = np.exp(np.sum(-0.5 * (stack ** 2), axis=2))
        return self.kernel_matrix

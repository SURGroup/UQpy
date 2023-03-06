from typing import Union

import numpy as np

from UQpy.utilities.kernels.baseclass.EuclideanKernel import EuclideanKernel
from scipy.spatial.distance import cdist
from scipy.special import gamma, kv


class Matern(EuclideanKernel):
    def __init__(self, kernel_parameter: Union[int, float] = 1, nu=1.5):
        """
        Matern Kernel is a generalization of Radial Basis Function kernel.

        :params nu: Shape parameter. For nu=0.5, 1.5, 2.5 and infinity, matern coincides with the exponential,
         matern-3/2, matern-5/2 and RBF covariance function, respectively.
        """
        super().__init__(kernel_parameter)
        self.nu = nu

    def calculate_kernel_matrix(self, x, s):
        l = self.kernel_parameter
        stack = cdist(x / l, s / l, metric='euclidean')
        if self.nu == 0.5:
            self.kernel_matrix = np.exp(-np.abs(stack))
        elif self.nu == 1.5:
            self.kernel_matrix = (1 + np.sqrt(3) * stack) * np.exp(-np.sqrt(3) * stack)
        elif self.nu == 2.5:
            self.kernel_matrix = (1 + np.sqrt(5) * stack + 5 * (stack ** 2) / 3) * np.exp(-np.sqrt(5) * stack)
        elif self.nu == np.inf:
            self.kernel_matrix = np.exp(-(stack ** 2) / 2)
        else:
            stack *= np.sqrt(2 * self.nu)
            tmp = 1 / (gamma(self.nu) * (2 ** (self.nu - 1)))
            tmp1 = stack ** self.nu
            tmp2 = kv(self.nu, stack)
            self.kernel_matrix = tmp * tmp1 * tmp2
        return self.kernel_matrix

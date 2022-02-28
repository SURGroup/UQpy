import numpy as np
from UQpy.utilities.kernels import EuclideanKernel
import scipy.spatial.distance as sd


class GaussianKernel(EuclideanKernel):
    """
    A class to calculate the Gaussian kernel defined as:

    .. math::
        k(x_j, x_i) = \exp[-(x_j - xj)^2/4\epsilon]

    """
    def __init__(self, epsilon: float = 1.0):
        """
        :param epsilon: Scale parameter of the Gaussian kernel
        """
        self.epsilon = epsilon

    def kernel_entry(self, xi, xj):
        return np.linalg.norm(xi - xj, "fro") ** 2

    def kernel_function(self, distance_pairs):
        return np.exp(-sd.squareform(distance_pairs) / (4 * self.epsilon))

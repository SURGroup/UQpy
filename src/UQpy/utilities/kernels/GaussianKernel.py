import itertools
from typing import Tuple

import numpy as np
import scipy
from scipy.spatial.distance import cdist

from UQpy.utilities.ValidationTypes import RandomStateType, Numpy2DFloatArray
from UQpy.utilities.kernels import EuclideanKernel
from scipy.spatial.distance import pdist
import scipy.spatial.distance as sd


class GaussianKernel(EuclideanKernel):
    """
    A class to calculate the Gaussian kernel defined as:

    .. math::
        k(x_j, x_i) = \exp[-(x_j - xj)^2/4\epsilon]

    """
    def __init__(self, kernel_parameter: float = 1.0):
        """
        :param epsilon: Scale parameter of the Gaussian kernel
        """
        super().__init__(kernel_parameter=kernel_parameter)

    def calculate_kernel_matrix(self, x, s):
        product = [self.element_wise_operation(point_pair)
                   for point_pair in list(itertools.product(x, s))]
        self.kernel_matrix = np.array(product).reshape(len(x), len(s))
        return self.kernel_matrix

    def element_wise_operation(self, xi_j: Tuple) -> float:
        xi, xj = xi_j

        if len(xi.shape) == 1:
            d = pdist(np.array([xi, xj]), "sqeuclidean")
        else:
            d = np.linalg.norm(xi - xj, 'fro') ** 2
        return np.exp(-d / (2 * self.kernel_parameter ** 2))

    def optimize_parameters(self, data: np.ndarray, tolerance: float,
                            n_nearest_neighbors: int,
                            n_cutoff_samples: int,
                            random_state: RandomStateType = None):
        """

        :param data: Set of data points.
        :param tolerance: Tolerance below which the Gaussian kernel is assumed to be zero.
        :param n_nearest_neighbors: Number of neighbors to use for cut-off estimation.
        :param n_cutoff_samples: Number of samples to use for cut-off estimation.
        :param random_state: Random seed used to initialize the pseudo-random number generator. If an :any:`int` is
         provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise, the
         object itself can be passed directly.
        """

        cut_off = self._estimate_cut_off(data, n_nearest_neighbors, n_cutoff_samples, random_state)
        self.epsilon = cut_off ** 2 / (-np.log(tolerance))

    def _estimate_cut_off(self, data, n_nearest_neighbors, n_partition, random_state):
        data = np.atleast_2d(data)
        n_points = data.shape[0]
        if n_points < 10:
            d = scipy.spatial.distance.pdist(data)
            return np.max(d)

        if n_partition is not None:
            random_indices = np.random.default_rng(random_state).permutation(n_points)
            distance_matrix = sd.cdist(data[random_indices[:n_partition]], data, metric='euclidean')
        else:
            distance_matrix = sd.squareform(sd.pdist(data, metric='euclidean'))
        k = np.min([n_nearest_neighbors, distance_matrix.shape[1]])
        k_smallest_values = np.partition(distance_matrix, k - 1, axis=1)[:, k - 1]

        est_cutoff = np.max(k_smallest_values)
        return float(est_cutoff)

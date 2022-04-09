import numpy as np
import scipy
import scipy.spatial.distance as sd

from UQpy.utilities.ValidationTypes import RandomStateType, Numpy2DFloatArray
from UQpy.utilities.kernels import EuclideanKernel
from scipy.spatial.distance import pdist


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
        super().__init__()
        self.epsilon = epsilon

    def kernel_entry(self, xi: Numpy2DFloatArray, xj: Numpy2DFloatArray):
        """
        Given two points, this method computes the Gaussian kernel value between those two points

        :param xi: First point.
        :param xj: Second point.
        :return: Float representing the kernel entry.
        """
        if len(xi.shape) == 1:
            d = pdist(np.array([xi, xj]), "sqeuclidean")
        else:
            d = np.linalg.norm(xi-xj, 'fro') ** 2
        return np.exp(-d / (2*self.epsilon**2))

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

import itertools

import numpy as np
import scipy.spatial.distance as sd

from UQpy.dimension_reduction.kernels.baseclass.Kernel import Kernel


class GaussianKernel(Kernel):
    """
    A class to calculate the Gaussian kernel defined as:

    .. math::
        k(x_j, x_i) = \exp[-(x_j - xj)^2/4\epsilon]

    """
    def __init__(self, epsilon=None):
        """

        :param float epsilon: Scale parameter of the Gaussian kernel
        """
        self.epsilon = epsilon

    def apply_method(self, data):
        pass

    def kernel_entry(self, xi, xj):
        pass

    @staticmethod
    def compute_default_epsilon(distance_pairs):
        """
        Compute a suitable epsilon when it is not provided by the user.
        Compute epsilon as the median of the square of the euclidean distances
        :param list distance_pairs:
        :rtype float
        """
        epsilon = np.median(np.array(distance_pairs) ** 2)
        return epsilon

    def kernel_operator(self, points, p=None):
        """
        Compute the Gaussian kernel entry for two points on the Euclidean space.
        :param list or numpy.array points:
        :rtype np.ndarray
        """
        distance_pairs = None
        if len(np.shape(points)) == 2:
            distance_pairs = sd.pdist(points, "euclidean")

        elif len(np.shape(points)) == 3:
            nargs = len(points)
            indices = range(nargs)
            pairs = list(itertools.combinations(indices, 2))
            distance_pairs = list()
            for id_pair in range(np.shape(pairs)[0]):
                i = pairs[id_pair][0]
                j = pairs[id_pair][1]

                xi = points[i]
                xj = points[j]

                distance_pairs.append(np.linalg.norm(xi - xj, "fro"))

        if self.epsilon is None:
            self.epsilon = self.compute_default_epsilon(distance_pairs)

        kernel = np.exp(-sd.squareform(distance_pairs) ** 2 / (4 * self.epsilon))

        return kernel

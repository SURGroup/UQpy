import itertools
from typing import Union

import numpy as np
import scipy.spatial.distance as sd
from UQpy.utilities.ValidationTypes import NumpyFloatArray
from UQpy.utilities.kernels import EuclideanKernel


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

    def apply_method(self, data):
        pass

    def kernel_entry(self, xi, xj):
        pass

    def kernel_operator(self, points: Union[list, NumpyFloatArray], **kwargs) -> NumpyFloatArray:
        """
        Compute the Gaussian kernel entry for two points on the Euclidean space.

        :param points: Coordinates of the points in the Euclidean space
        :param p:

        """
        distance_pairs = None
        if len(np.shape(points)) == 2:
            distance_pairs = sd.pdist(points, "sqeuclidean")

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

                distance_pairs.append(np.linalg.norm(xi - xj, "fro") ** 2)

        kernel = np.exp(-sd.squareform(distance_pairs) / (4 * self.epsilon))

        return kernel

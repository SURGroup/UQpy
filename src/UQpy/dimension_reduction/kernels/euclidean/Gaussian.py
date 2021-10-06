import itertools

import numpy as np
import scipy.spatial.distance as sd

from UQpy.dimension_reduction.kernels.grassmanian.baseclass.Kernel import Kernel


class Gaussian(Kernel):

    def __init__(self, epsilon: float = None):
        self.epsilon = epsilon

    def kernel_function(self, distance_pairs):
        return np.exp(-sd.squareform(distance_pairs) ** 2 / (4 * self.epsilon))

    def compute_default_epsilon(self, epsilon):
        if epsilon is None:
            # Compute a suitable epsilon when it is not provided by the user.
            # Compute epsilon as the median of the square of the euclidean distances
            epsilon = np.median(np.array(self.distance_pairs) ** 2)
        return epsilon

    def apply_method(self, data):
        self.distance_pairs = []
        if len(np.shape(data)) == 2:
            # Set of 1-D arrays
            self.distance_pairs = sd.pdist(data, 'euclidean')
        elif len(np.shape(data)) == 3:
            # Set of 2-D arrays
            # Check arguments: verify the consistency of input arguments.
            nargs = len(data)
            indices = range(nargs)
            pairs = list(itertools.combinations(indices, 2))

            self.distance_pairs = []
            for id_pair in range(np.shape(pairs)[0]):
                ii = pairs[id_pair][0]  # Point i
                jj = pairs[id_pair][1]  # Point j

                x0 = data[ii]
                x1 = data[jj]

                distance = np.linalg.norm(x0 - x1, 'fro')

                self.distance_pairs.append(distance)
        else:
            raise TypeError('UQpy: The size of the input data is not consistent with this method.')

        self.epsilon = self.compute_default_epsilon(self.epsilon)

        return self.kernel_function(distance_pairs=self.distance_pairs)

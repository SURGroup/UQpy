import numpy as np
import scipy.spatial.distance as sd


class Gaussian:

    def apply_kernel(self, distance_pairs, epsilon):
        return np.exp(-sd.squareform(distance_pairs) ** 2 / (4 * epsilon))

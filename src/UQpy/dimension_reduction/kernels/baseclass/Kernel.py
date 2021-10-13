import itertools
from abc import ABC, abstractmethod
import numpy as np
import scipy.spatial.distance as sd


class Kernel(ABC):
    @abstractmethod
    def apply_method(self, data):
        pass

    def kernel_operator(self, points, p_dim):

        # Check points for type and shape consistency.
        # -----------------------------------------------------------
        if not isinstance(points, list) and not isinstance(points, np.ndarray):
            raise TypeError("UQpy: `points` must be either list or numpy.ndarray.")

        nargs = len(points)

        if nargs < 2:
            raise ValueError("UQpy: At least two matrices must be provided.")
        # ------------------------------------------------------------

        # Define the pairs of points to compute the entries of the kernel matrix.
        indices = range(nargs)
        pairs = list(itertools.combinations(indices, 2))

        # Estimate off-diagonal entries of the kernel matrix.
        kernel_list = []
        for id_pair in range(np.shape(pairs)[0]):
            ii = pairs[id_pair][0]  # Point i
            jj = pairs[id_pair][1]  # Point j

            x0 = np.asarray(points[ii])[:, :p_dim]
            x1 = np.asarray(points[jj])[:, :p_dim]

            ker = self.pointwise_operator(x0, x1)
            kernel_list.append(ker)

        # Diagonal entries of the kernel matrix.
        kernel_diag = []
        for id_elem in range(nargs):
            xd = np.asarray(points[id_elem])
            xd = xd[:, :p_dim]

            kerd = self.pointwise_operator(xd, xd)
            kernel_diag.append(kerd)

        # Add the diagonals and off-diagonal entries of the Kernel matrix.
        kernel_matrix = sd.squareform(np.array(kernel_list)) + np.diag(kernel_diag)

        # Return the kernel matrix.
        return kernel_matrix

    @abstractmethod
    def pointwise_operator(self, point1, point2):
        pass

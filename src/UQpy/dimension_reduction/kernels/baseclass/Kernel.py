import itertools
from abc import ABC, abstractmethod
import numpy as np

from UQpy.dimension_reduction.grassmann_manifold.GrassmannPoint import GrassmannPoint


class Kernel(ABC):
    @abstractmethod
    def apply_method(self, data):
        pass

    @abstractmethod
    def kernel_entry(self, xi, xj):
        pass

    def kernel_operator(self, points: list[GrassmannPoint], p: int = None):
        nargs = len(points)
        # Define the pairs of points to compute the entries of the kernel matrix.
        indices = range(nargs)
        pairs = list(itertools.combinations_with_replacement(indices, 2))

        # Estimate entries of the kernel matrix.
        kernel = np.zeros((nargs, nargs))
        for id_pair in range(np.shape(pairs)[0]):
            i = pairs[id_pair][0]  # Point i
            j = pairs[id_pair][1]  # Point j
            if not p:
                xi = points[i]
                xj = points[j]
            else:
                xi = GrassmannPoint(points[i].data[:, :p])
                xj = GrassmannPoint(points[j].data[:, :p])

            # RiemannianDistance.check_rows(xi, xj)
            kernel[i, j] = self.kernel_entry(xi, xj)
            kernel[j, i] = kernel[i, j]

        return kernel


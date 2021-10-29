import itertools
from abc import ABC, abstractmethod
import numpy as np

from UQpy.dimension_reduction.distances.grassmanian.baseclass import RiemannianDistance


class Kernel(ABC):
    @abstractmethod
    def apply_method(self, data):
        pass

    @abstractmethod
    def kernel_entry(self, xi, xj):
        pass

    @staticmethod
    def check_data(points):
        if not isinstance(points, list) and not isinstance(points, np.ndarray):
            raise TypeError("UQpy: Data points must be provided either as a list or a numpy.ndarray.")
        elif isinstance(points, np.ndarray):
            if len(points.shape) != 3:
                raise TypeError("UQpy: Data points must be provided as a 3D numpy.ndarray.")
            else:
                nargs = points.shape[0]
        else:
            nargs = len(points)
        if nargs < 2:
            raise ValueError("UQpy: At least two matrices must be provided.")

        return nargs

    def kernel_operator(self, points, p=None):

        nargs = Kernel.check_data(points)
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
                xi = points[i][:, :p]
                xj = points[j][:, :p]

            RiemannianDistance.check_points(xi, xj)
            kernel[i, j] = self.kernel_entry(xi, xj)
            kernel[j, i] = kernel[i, j]

        return kernel


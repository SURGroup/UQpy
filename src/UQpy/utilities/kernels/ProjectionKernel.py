import numpy as np

from UQpy.utilities.GrassmannPoint import GrassmannPoint
from UQpy.utilities.kernels import GrassmannianKernel


class ProjectionKernel(GrassmannianKernel):
    """
    A class to calculate the Projection kernel

    """
    def apply_method(self, points):
        points.evaluate_matrix(self, self.calculate_kernel_matrix)

    def kernel_entry(self, xi: GrassmannPoint, xj: GrassmannPoint) -> float:
        """
        Compute the Projection kernel entry for two points on the Grassmann manifold.

        :param xi: Orthonormal matrix representing the first point.
        :param xj: Orthonormal matrix representing the second point.
        """
        r = np.dot(xi.data.T, xj.data)
        n = np.linalg.norm(r, "fro")
        kij = n * n
        return kij

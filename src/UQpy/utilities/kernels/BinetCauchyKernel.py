import numpy as np

from UQpy.utilities.GrassmannPoint import GrassmannPoint
from UQpy.utilities.kernels import GrassmannianKernel


class BinetCauchyKernel(GrassmannianKernel):
    """
    A class to calculate the Binet-Cauchy kernel.

    """
    def apply_method(self, points):
        points.evaluate_matrix(self, self.calculate_kernel_matrix)

    def kernel_entry(self, xi: GrassmannPoint, xj: GrassmannPoint) -> float:
        """
        Compute the Binet-Cauchy kernel entry for two points on the Grassmann manifold.

        :param xi: Orthonormal matrix representing the first point.
        :param xj: Orthonormal matrix representing the second point.

        """
        r = np.dot(xi.data.T, xj.data)
        det = np.linalg.det(r)
        return det * det

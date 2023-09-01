from typing import Tuple

import numpy as np

from UQpy.utilities.kernels import GrassmannianKernel


class BinetCauchyKernel(GrassmannianKernel):
    """
    A class to calculate the Binet-Cauchy kernel.

    """
    def element_wise_operation(self, xi_j: Tuple) -> float:
        """
        Compute the Projection kernel entry for a tuple of points on the Grassmann manifold.

        :param xi_j: Tuple of orthonormal matrices representing the grassmann points.
        """
        xi, xj = xi_j
        r = np.dot(xi.T, xj)
        det = np.linalg.det(r)
        return det * det

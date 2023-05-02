from typing import Union, Tuple

import numpy as np

from UQpy.utilities.kernels.baseclass.GrassmannianKernel import GrassmannianKernel


class ProjectionKernel(GrassmannianKernel):

    def __init__(self, kernel_parameter: Union[int, float] = None):
        """
        :param kernel_parameter: Number of independent p-planes of each Grassmann point.
        """
        super().__init__(kernel_parameter)

    def element_wise_operation(self, xi_j: Tuple) -> float:
        """
        Compute the Projection kernel entry for a tuple of points on the Grassmann manifold.

        :param xi_j: Tuple of orthonormal matrices representing the grassmann points.
        """
        xi, xj = xi_j
        r = np.dot(xi.T, xj)
        n = np.linalg.norm(r, "fro")
        return n * n

import numpy as np
from line_profiler_pycharm import profile

from UQpy.dimension_reduction.grassmann_manifold.GrassmannPoint import GrassmannPoint
from UQpy.dimension_reduction.kernels.baseclass.Kernel import Kernel


class ProjectionKernel(Kernel):
    """
    A class to calculate the Projection kernel defined as:

    .. math::

        k_p(x_j, x_i) = (||x_j'\cdot xj||_F)^2

    """
    def apply_method(self, points):
        points.evaluate_matrix(self, self.kernel_operator)

    @profile
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

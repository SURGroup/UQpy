import numpy as np
from UQpy.dimension_reduction.kernels.baseclass.Kernel import Kernel


class ProjectionKernel(Kernel):
    """
    A class to calculate the Projection kernel defined as:

    .. math::

        k_p(x_j, x_i) = (||x_j'\cdot xj||_F)^2

    """
    def apply_method(self, points):
        points.evaluate_matrix(self, self.kernel_operator)

    def kernel_entry(self, xi, xj):
        """
        Compute the Projection kernel entry for two points on the Grassmann manifold.

        :param numpy.array xi: Orthonormal matrix representing the first subspace.
        :param numpy.array xj: Orthonormal matrix representing the second subspace.
        :rtype: float
        """
        r = np.dot(xi.T, xj)
        n = np.linalg.norm(r, "fro")
        kij = n * n
        return kij

import numpy as np

from UQpy.dimension_reduction.kernels.baseclass.Kernel import Kernel


class BinetCauchyKernel(Kernel):
    """
    A class to calculate the Binet-Cauchy kernel defined as:

    .. math::

        k_p(x_j, x_i) = det(x_j'\cdot xj)^2

    """
    def apply_method(self, points):
        points.evaluate_matrix(self, self.kernel_operator)

    def kernel_entry(self, xi, xj):
        """
        Compute the Binet-Cauchy kernel entry for two points on the Grassmann manifold.

        :param numpy.array xi: Orthonormal matrix representing the first subspace.
        :param numpy.array xj: Orthonormal matrix representing the second subspace.
        :rtype: float
        """
        r = np.dot(xi.T, xj)
        det = np.linalg.det(r)
        kij = det * det
        return kij

import numpy as np

from UQpy.dimension_reduction.distances.grassmann.baseclass.RiemannianDistance import (
    RiemannianDistance,
)


class BinetCauchyDistance(RiemannianDistance):
    """
    A class to calculate the Binet-Cauchy distance between two  Grassmann points defined as:

    .. math::

        d_{BC}(x_i, x_j) = [1-\prod_{l}\cos^2(\Theta_l)]^{1/2}

    """

    def compute_distance(self, xi, xj) -> float:
        """
        Compute the Binet-Cauchy distance between two points on the Grassmann manifold

        :param numpy.array xi: Orthonormal matrix representing the first point.
        :param numpy.array xj: Orthonormal matrix representing the second point.
        :rtype: float
        """
        RiemannianDistance.check_points(xi, xj)

        r = np.dot(xi.T, xj)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        si[np.where(si > 1)] = 1.0
        theta = np.arccos(si)

        cos_sq = np.cos(theta) ** 2
        d = np.sqrt(1 - np.prod(cos_sq))

        return d

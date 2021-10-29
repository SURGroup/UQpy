import numpy as np

from UQpy.dimension_reduction.distances.grassmanian.baseclass.RiemannianDistance import (
    RiemannianDistance,
)


class FubiniStudyDistance(RiemannianDistance):
    """
    A class to calculate the Fubini-Study distance between two  Grassmann points defined as:

    .. math::
        x_j' x_i = UΣV

        \Theta = cos^{-1}(Σ)

        d_{C}(x_i, x_j) = cos^{-1}(\prod_{l}\cos(\Theta_l))

    """
    def compute_distance(self, xi, xj) -> float:
        """
        Compute the Fubini-Study distance between two points on the Grassmann manifold
        :param numpy.array xi: Orthonormal matrix representing the first point.
        :param numpy.array xj: Orthonormal matrix representing the first point.
        :rtype float
        """
        RiemannianDistance.check_points(xi, xj)

        r = np.dot(xi.T, xj)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(si)
        cos_t = np.cos(theta)
        d = np.arccos(np.prod(cos_t))

        return d

import numpy as np

from UQpy.dimension_reduction.distances.grassmanian.baseclass.RiemannianDistance import (
    RiemannianDistance,
)


class ProcrustesDistance(RiemannianDistance):
    """
    A class to calculate the Procrustes (chordal) distance between two  Grassmann points defined as:

    .. math::
        x_j' x_i = UΣV

        \Theta = cos^{-1}(Σ)

        d_{C}(x_i, x_j) = 2[\sum_{l}\sin^2(\Theta_l/2)]^{1/2}

    """
    def compute_distance(self, xi, xj) -> float:
        """
        Compute the Procrustes (chordal) distance between two points on the Grassmann manifold
        :param numpy.array xi: Orthonormal matrix representing the first point.
        :param numpy.array xj: Orthonormal matrix representing the first point.
        :rtype float
        """
        RiemannianDistance.check_points(xi, xj)

        rank_i = xi.shape[1]
        rank_j = xj.shape[1]

        r = np.dot(xi.T, xj)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        si[np.where(si > 1)] = 1.0
        theta = np.arccos(si)
        sin_sq = np.sin(theta / 2) ** 2
        d = 2 * np.sqrt(abs(rank_i - rank_j) + np.sum(sin_sq))

        return d

import numpy as np

from UQpy.dimension_reduction.distances.grassmanian.baseclass.RiemannianDistance import (
    RiemannianDistance,
)


class Projection(RiemannianDistance):
    """
    A class to calculate the Projection distance between two Grassmann points defined as:

    .. math::
        x_j' x_i = UΣV

        \Theta = cos^{-1}(Σ)

        d_{C}(x_i, x_j) =  (\sum_{l} \sin^2(\Theta_l))^{1/2}

    """
    def compute_distance(self, xi, xj) -> float:
        """
        Compute the Projection distance between two points on the Grassmann manifold
        :param numpy.array xi: Orthonormal matrix representing the first point.
        :param numpy.array xj: Orthonormal matrix representing the first point.
        :rtype float
        """
        RiemannianDistance.check_points(xi, xj)

        r = np.dot(xi.T, xj)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        si[np.where(si > 1)] = 1.0
        theta = np.arccos(si)
        d = np.sqrt(np.sum(np.sin(theta) ** 2))

        return d

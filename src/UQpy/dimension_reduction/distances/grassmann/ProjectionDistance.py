import numpy as np

from UQpy.dimension_reduction.distances.grassmann.baseclass.RiemannianDistance import (
    RiemannianDistance,
)


class ProjectionDistance(RiemannianDistance):
    """
    A class to calculate the Projection distance between two Grassmann points defined as:

    .. math::

        d_{C}(x_i, x_j) =  (\sum_{l} \sin^2(\Theta_l))^{1/2}

    """
    def compute_distance(self, xi, xj) -> float:
        """
        Compute the Projection distance between two points on the Grassmann manifold.

        :param numpy.array xi: Orthonormal matrix representing the first subspace.
        :param numpy.array xj: Orthonormal matrix representing the second subspace.
        :rtype: float
        """
        RiemannianDistance.check_points(xi, xj)

        rank_i = xi.shape[1]
        rank_j = xj.shape[1]

        r = np.dot(xi.T, xj)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        si[np.where(si > 1)] = 1.0
        theta = np.arccos(si)
        d = np.sqrt(abs(rank_i - rank_j) + np.sum(np.sin(theta) ** 2))

        return d

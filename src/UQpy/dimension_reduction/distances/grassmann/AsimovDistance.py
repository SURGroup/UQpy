import numpy as np

from UQpy.dimension_reduction.distances.grassmann.baseclass.RiemannianDistance import (
    RiemannianDistance,
)


class AsimovDistance(RiemannianDistance):
    """
    A class to calculate the Asimov distance between two  Grassmann points defined as:

    .. math::

        d_{A}(x_i, x_j) = \max(\Theta)

    """

    def compute_distance(self, xi, xj) -> float:
        """
        Compute the Asimov distance between two points on the Grassmann manifold.

        :param numpy.array xi: Orthonormal matrix representing the first subspace.
        :param numpy.array xj: Orthonormal matrix representing the second subspace.
        :rtype: float
        """
        RiemannianDistance.check_points(xi, xj)

        r = np.dot(xi.T, xj)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        si[np.where(si > 1)] = 1.0
        theta = np.arccos(si)
        d = np.max(theta)

        return d

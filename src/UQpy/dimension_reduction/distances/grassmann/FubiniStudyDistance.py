import numpy as np

from UQpy.dimension_reduction.distances.grassmann.baseclass.RiemannianDistance import (
    RiemannianDistance,
)
from UQpy.dimension_reduction.grassmann_manifold.GrassmannPoint import GrassmannPoint


class FubiniStudyDistance(RiemannianDistance):
    """
    A class to calculate the Fubini-Study distance between two  Grassmann points defined as:

    .. math::

        d_{C}(x_i, x_j) = cos^{-1}(\prod_{l}\cos(\Theta_l))

    """
    def compute_distance(self, xi: GrassmannPoint, xj: GrassmannPoint) -> float:
        """
        Compute the Fubini-Study distance between two points on the Grassmann manifold.

        :param numpy.array xi: Orthonormal matrix representing the first subspace.
        :param numpy.array xj: Orthonormal matrix representing the second subspace.
        :rtype: float
        """
        RiemannianDistance.check_rows(xi, xj)

        r = np.dot(xi.data.T, xj.data)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(si)
        cos_t = np.cos(theta)
        distance = np.arccos(np.prod(cos_t))

        return distance

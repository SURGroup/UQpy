import numpy as np
from beartype import beartype

from UQpy.dimension_reduction.distances.baseclass.RiemannianDistance import (
    RiemannianDistance,
)
from UQpy.dimension_reduction.grassmann_manifold.GrassmannPoint import GrassmannPoint


class AsimovDistance(RiemannianDistance):
    """
    A class to calculate the Asimov distance between two  Grassmann points defined as:

    .. math::

        d_{A}(x_i, x_j) = \max(\Theta)

    """
    @beartype
    def compute_distance(self, xi: GrassmannPoint, xj: GrassmannPoint) -> float:
        """
        Compute the Asimov distance between two points on the Grassmann manifold.

        :param GrassmannPoint xi: Orthonormal matrix representing the first point.
        :param GrassmannPoint xj: Orthonormal matrix representing the second point.
        :rtype: float
        """
        RiemannianDistance.check_rows(xi, xj)

        r = np.dot(xi.data.T, xj.data)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        si[np.where(si > 1)] = 1.0
        theta = np.arccos(si)
        distance = np.max(theta)

        return distance

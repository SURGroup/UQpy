import numpy as np
from beartype import beartype

from UQpy.utilities.distances.baseclass.GrassmannianDistance import (
    GrassmannianDistance,
)
from UQpy.utilities.GrassmannPoint import GrassmannPoint


class AsimovDistance(GrassmannianDistance):
    """
    A class to calculate the Asimov distance between two Grassmann points.

    """
    @beartype
    def compute_distance(self, xi: GrassmannPoint, xj: GrassmannPoint) -> float:
        """
        Compute the Asimov distance between two points on the Grassmann manifold.

        :param xi: Orthonormal matrix representing the first point.
        :param xj: Orthonormal matrix representing the second point.

        """
        GrassmannianDistance.check_rows(xi, xj)

        r = np.dot(xi.data.T, xj.data)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        si[np.where(si > 1)] = 1.0
        theta = np.arccos(si)
        distance = np.max(theta)

        return distance

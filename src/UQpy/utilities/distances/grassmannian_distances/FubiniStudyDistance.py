import numpy as np

from UQpy.utilities.distances.baseclass.GrassmannianDistance import (
    GrassmannianDistance,
)
from UQpy.utilities.GrassmannPoint import GrassmannPoint


class FubiniStudyDistance(GrassmannianDistance):
    """
    A class to calculate the Fubini-Study distance between two Grassmann points.

    """
    def compute_distance(self, xi: GrassmannPoint, xj: GrassmannPoint) -> float:
        """
        Compute the Fubini-Study distance between two points on the Grassmann manifold.

        :param xi: Orthonormal matrix representing the first point.
        :param xj: Orthonormal matrix representing the second point.

        """
        GrassmannianDistance.check_rows(xi, xj)

        r = np.dot(xi.data.T, xj.data)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(si)
        cos_t = np.cos(theta)
        distance = np.arccos(np.prod(cos_t))

        return distance

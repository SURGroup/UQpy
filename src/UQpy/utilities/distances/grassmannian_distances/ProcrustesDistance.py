import numpy as np

from UQpy.utilities.distances.baseclass.GrassmannianDistance import (
    GrassmannianDistance,
)
from UQpy.utilities.GrassmannPoint import GrassmannPoint


class ProcrustesDistance(GrassmannianDistance):
    """
    A class to calculate the Procrustes distance between two Grassmann points.

    """
    def compute_distance(self, xi: GrassmannPoint, xj: GrassmannPoint) -> float:
        """
        Compute the Procrustes distance between two points on the Grassmann manifold.

        :param xi: Orthonormal matrix representing the first point.
        :param xj: Orthonormal matrix representing the second point.

        """
        GrassmannianDistance.check_rows(xi, xj)

        rank_i = xi.data.shape[1]
        rank_j = xj.data.shape[1]

        r = np.dot(xi.data.T, xj.data)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        si[np.where(si > 1)] = 1.0
        theta = np.arccos(si)
        sin_sq = np.sin(theta / 2) ** 2
        distance = 2 * np.sqrt(abs(rank_i - rank_j) + np.sum(sin_sq))

        return distance

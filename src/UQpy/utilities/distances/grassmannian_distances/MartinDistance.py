import numpy as np
import sys
from UQpy.utilities.distances.baseclass.GrassmannianDistance import (
    GrassmannianDistance,
)
from UQpy.utilities.GrassmannPoint import GrassmannPoint


class MartinDistance(GrassmannianDistance):
    """
    A class to calculate the Martin distance between two Grassmann points.

    """
    def compute_distance(self, xi: GrassmannPoint, xj: GrassmannPoint) -> float:
        """
        Compute the Martin distance between two points on the Grassmann manifold.

        :param xi: Orthonormal matrix representing the first point.
        :param xj: Orthonormal matrix representing the second point.

        """
        GrassmannianDistance.check_rows(xi, xj)

        r = np.dot(xi.data.T, xj.data)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        si[np.where(si > 1)] = 1.0
        theta = np.arccos(si)
        cos_sq = np.cos(theta) ** 2
        float_min = sys.float_info.min
        index = np.where(cos_sq < float_min)
        cos_sq[index] = float_min
        recp = np.reciprocal(cos_sq)
        distance = np.sqrt(np.log(np.prod(recp)))

        return distance

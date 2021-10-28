from typing import Union

import numpy as np
from UQpy.dimension_reduction.distances.grassmanian.baseclass.RiemannianDistance import (
    RiemannianDistance,
)


class Geodesic(RiemannianDistance):
    """
    TODO: Add description of ArcLength distance (This should reflect on the documentation).
    TODO: Validate results.
    """

    def compute_distance(self, xi, xj) -> float:
        """
        Geodesic (or arc length or Grassmann) distance
        :param xi:
        :param xj:
        :return:
        """
        RiemannianDistance.check_points(xi, xj)

        rank_i = xi.shape[1]
        rank_j = xj.shape[1]

        r = np.dot(xi.T, xj)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        si[np.where(si > 1)] = 1.0
        theta = np.arccos(si)
        distance = (np.sqrt(abs(rank_i - rank_j) * np.pi ** 2 / 4 + np.sum(theta ** 2)))

        return distance

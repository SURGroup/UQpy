import numpy as np
from numpy.linalg import svd

from UQpy.dimension_reduction.distances.grassmanian.baseclass.RiemannianDistance import (
    RiemannianDistance,
)


class FubiniStudy(RiemannianDistance):
    def compute_distance(self, xi, xj) -> float:
        """
        Fubini Study distance
        :param xi:
        :param xj:
        :return:
        """
        RiemannianDistance.check_points(xi, xj)

        r = np.dot(xi.T, xj)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(si)
        cos_t = np.cos(theta)
        d = np.arccos(np.prod(cos_t))

        return d

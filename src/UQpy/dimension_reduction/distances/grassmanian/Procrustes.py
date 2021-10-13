import numpy as np
from numpy.linalg import svd

from UQpy.dimension_reduction.distances.grassmanian.baseclass.RiemannianDistance import (
    RiemannianDistance,
)


class Procrustes(RiemannianDistance):
    def compute_distance(self, point1, point2):

        point1, point2 = RiemannianDistance.check_points(point1, point2)

        l = min(np.shape(point1))
        k = min(np.shape(point2))
        rank = min(l, k)

        r = np.dot(point1.T, point2)
        (ui, si, vi) = svd(r, rank)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        sin_sq = np.sin(theta / 2) ** 2
        d = np.sqrt(abs(k - l) + 2 * np.sum(sin_sq))

        return d

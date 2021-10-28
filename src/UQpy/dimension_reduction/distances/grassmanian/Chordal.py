import numpy as np

from UQpy.dimension_reduction.distances.grassmanian.baseclass.RiemannianDistance import (
    RiemannianDistance,
)


class Chordal(RiemannianDistance):
    def compute_distance(self, xi, xj) -> float:
        """
        Chordal (or Procrustes distance)
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
        sin_sq = np.sin(theta / 2) ** 2
        d = np.sqrt(abs(rank_i - rank_j) + 2 * np.sum(sin_sq))

        return d

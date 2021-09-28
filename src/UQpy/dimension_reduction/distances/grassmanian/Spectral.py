import numpy as np
from numpy.linalg import svd

from UQpy.dimension_reduction.distances.grassmanian.baseclass.RiemannianDistance import RiemannianDistance


class Spectral(RiemannianDistance):

    def compute_distance(self, point1, point2):

        point1, point2 = RiemannianDistance.check_points(point1, point2)

        l = min(np.shape(point1))
        k = min(np.shape(point2))

        if l != k:
            raise NotImplementedError('UQpy: distance not implemented for manifolds with distinct dimensions.')

        rank = min(l, k)

        r = np.dot(point1.T, point2)
        (ui, si, vi) = svd(r, rank)

        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        d = 2 * np.sin(np.max(theta) / 2)

        return d

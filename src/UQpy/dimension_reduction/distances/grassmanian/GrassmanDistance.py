import numpy as np
from numpy.linalg import svd
from UQpy.dimension_reduction.distances.grassmanian.baseclass.RiemannianDistance import RiemannianDistance


class GrassmannDistance(RiemannianDistance):

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
        distance = np.sqrt(abs(k - l) * np.pi ** 2 / 4 + np.sum(theta ** 2))

        return distance

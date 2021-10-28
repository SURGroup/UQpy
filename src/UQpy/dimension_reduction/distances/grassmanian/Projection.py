import numpy as np

from UQpy.dimension_reduction.distances.grassmanian.baseclass.RiemannianDistance import (
    RiemannianDistance,
)


class Projection(RiemannianDistance):
    def compute_distance(self, xi, xj) -> float:

        RiemannianDistance.check_points(xi, xj)

        r = np.dot(xi.T, xj)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        si[np.where(si > 1)] = 1.0
        theta = np.arccos(si)
        d = np.sum(np.sin(theta) ** 2)

        return d

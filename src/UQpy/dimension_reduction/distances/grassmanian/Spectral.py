import numpy as np

from UQpy.dimension_reduction.distances.grassmanian.baseclass.RiemannianDistance import (
    RiemannianDistance,
)


class Spectral(RiemannianDistance):
    def compute_distance(self, xi, xj) -> float:
        RiemannianDistance.check_points(xi, xj)

        r = np.dot(xi.T, xj)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        si[np.where(si > 1)] = 1.0
        theta = np.arccos(si)
        d = 2 * np.sin(np.max(theta) / 2)

        return d

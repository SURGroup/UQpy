import numpy as np
from UQpy.dimension_reduction.distances.grassmanian.baseclass.RiemannianDistance import (
    RiemannianDistance,
)


class ArcLength(RiemannianDistance):
    """
    TODO: Add description of ArcLength distance (This should reflect on the documentation).
    TODO: Validate results.
    """

    def compute_distance(self, point1, point2) -> float:
        """
        TODO: Add description of compute_distance method.
        """

        point1, point2 = RiemannianDistance.check_points(point1, point2)

        rank1 = min(np.shape(point1))
        rank2 = min(np.shape(point2))
        rank = min(rank1, rank2)

        r = np.dot(point1.T, point2)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        si[np.where(si > 1)] = 1.0
        theta = np.arccos(np.diag(si))
        d = np.sqrt(abs(rank1 - rank2) * np.pi ** 2 / 4 + np.sum(theta ** 2))

        return d

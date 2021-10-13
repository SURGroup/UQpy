from typing import Union

import numpy as np

from UQpy.dimension_reduction.distances.grassmanian.baseclass.RiemannianDistance import (
    RiemannianDistance,
)


class Asimov(RiemannianDistance):
    """

    """

    def compute_distance(self, point1: Union[list, float], point2) -> float:
        point1, point2 = RiemannianDistance.check_points(point1, point2)

        rank1 = min(np.shape(point1))
        rank2 = min(np.shape(point2))

        if rank1 != rank2:
            raise NotImplementedError(
                "UQpy: distance not implemented for manifolds with distinct dimensions."
            )

        r = np.dot(point1.T, point2)
        (ui, si, vi) = np.linalg.svd(r, rank2)

        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        d = np.max(theta)

        return d

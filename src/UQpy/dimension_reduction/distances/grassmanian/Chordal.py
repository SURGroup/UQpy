import numpy as np
from numpy.linalg import svd
from UQpy.dimension_reduction_v4.kernel_based.distances.baseclass.RiemannianDistance import RiemannianDistance, \
    check_points


class Chordal(RiemannianDistance):

    def compute_distance(self, point1, point2):

        """
        Chordal distance.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Chordal distance between x0 and x1.

        """

        point1, point2 = check_points(point1, point2)

        l = min(np.shape(point1))
        k = min(np.shape(point2))
        rank = min(l, k)

        r_star = np.dot(point2.T, point2)
        (ui, si, vi) = svd(r_star, rank)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        sin_sq = np.sin(theta) ** 2
        d = np.sqrt(abs(k - l) + np.sum(sin_sq))

        return d

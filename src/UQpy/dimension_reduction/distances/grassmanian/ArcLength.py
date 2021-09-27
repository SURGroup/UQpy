import numpy as np
from numpy.linalg import svd
from UQpy.dimension_reduction_v4.kernel_based.distances.baseclass.RiemannianDistance import RiemannianDistance, \
    check_points


class ArcLength(RiemannianDistance):

    def compute_distance(self, point1, point2):

        """
        Grassmann distance.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Grassmann distance between x0 and x1.

        """

        point1, point2 = check_points(point1, point2)

        l = min(np.shape(point1))
        k = min(np.shape(point2))
        rank = min(l, k)

        r = np.dot(point1.T, point2)
        (ui, si, vi) = svd(r, rank)

        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        d = np.sqrt(abs(k - l) * np.pi ** 2 / 4 + np.sum(theta ** 2))

        return d



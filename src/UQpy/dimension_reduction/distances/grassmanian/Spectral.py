import numpy as np
from numpy.linalg import svd
from UQpy.dimension_reduction_v4.kernel_based.distances.baseclass.RiemannianDistance import RiemannianDistance, \
    check_points


class Spectral(RiemannianDistance):

    def compute_distance(self, point1, point2):

        """
        Estimate the Binet-Cauchy distance.

        One of the distances defined on the Grassmann manifold is the projection distance.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Projection distance between x0 and x1.
        """

        point1, point2 = check_points(point1, point2)

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

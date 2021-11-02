import numpy as np

from UQpy.dimension_reduction.distances.grassmann.baseclass.RiemannianDistance import (
    RiemannianDistance,
)


class SpectralDistance(RiemannianDistance):
    """
    A class to calculate the Projection distance between two Grassmann points defined as:

    .. math::

        d_{C}(x_i, x_j) =  2\sin( \max(\Theta_l)/2)

    """
    def compute_distance(self, xi, xj) -> float:
        """
        Compute the Spectral distance between two points on the Grassmann manifold.
        :param numpy.array xi: Orthonormal matrix representing the first subspace.
        :param numpy.array xj: Orthonormal matrix representing the second subspace.
        :rtype: float
        """
        RiemannianDistance.check_points(xi, xj)

        r = np.dot(xi.T, xj)
        (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
        si[np.where(si > 1)] = 1.0
        theta = np.arccos(si)
        d = 2 * np.sin(np.max(theta) / 2)

        return d

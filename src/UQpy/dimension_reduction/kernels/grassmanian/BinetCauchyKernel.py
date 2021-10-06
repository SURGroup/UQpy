import numpy as np

from UQpy.dimension_reduction.kernels.grassmanian.baseclass.Kernel import Kernel


class BinetCauchy(Kernel):

    def apply_method(self, data):
        data.evaluate_matrix(self, self.kernel_operator)

    def pointwise_operator(self, point1, point2):
        if not isinstance(point1, list) and not isinstance(point1, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            point1 = np.array(point1)

        if not isinstance(point2, list) and not isinstance(point2, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            point2 = np.array(point2)

        r = np.dot(point1.T, point2)
        det = np.linalg.det(r)
        ker = det * det
        return ker

from UQpy.dimension_reduction_v4.kernel_based.kernels.kernel_methods.baseclass.KernelMethod import KernelMethod
import numpy as np


class BinetCauchy(KernelMethod):
    def apply_method(self, point1, point2):

        """
        Estimate the value of the Binet-Cauchy kernel between x0 and x1.

        One of the kernels defined on a manifold is the Binet-Cauchy kernel.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the grassman manifold.

        * **x1** (`list` or `ndarray`)
            Point on the grassman manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Kernel value for x0 and x1.

        """

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

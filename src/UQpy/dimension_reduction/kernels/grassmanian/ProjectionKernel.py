import numpy as np


class ProjectionKernel:

    def apply_method(self, point1, point2):
        """
        Estimate the value of the projection kernel between x0 and x1.

        One of the kernels defined on a manifold is the projection kernel.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the grassman manifold.

        * **x1** (`list` or `ndarray`)
            Point on the grassman manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Kernel value for x0 and x1.

        """

        if not isinstance(point1, list) and not isinstance(point2, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            point1 = np.array(point1)

        if not isinstance(point2, list) and not isinstance(point2, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            point2 = np.array(point2)

        r = np.dot(point1.T, point2)
        n = np.linalg.norm(r, 'fro')
        ker = n * n
        return ker

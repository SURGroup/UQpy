import numpy as np

def my_kernel(x0, x1):

    """
    User defined kernel.

    **Input:**

    * **x0** (`list` or `ndarray`)
        Point on the Grassmann manifold.

    * **x1** (`list` or `ndarray`)
        Point on the Grassmann manifold.

    **Output/Returns:**

    * **distance** (`float`)
        Kernel value for x0 and x1.
    """

    if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
        raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
    else:
        x0 = np.array(x0)

    if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
        raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
    else:
        x1 = np.array(x1)

    r = np.dot(x0.T, x1)
    det = np.linalg.det(r)
    ker = det * det
    return ker
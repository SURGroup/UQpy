"""

Auxiliary file
==============================================

"""

import numpy as np


def evaluate(X, params=[7, 0.1]):
    """Non-monotonic Ishigami-Homma three parameter test function"""

    a = params[0]
    b = params[1]

    Y = (
        np.sin(X[:, 0])
        + a * np.power(np.sin(X[:, 1]), 2)
        + b * np.power(X[:, 2], 4) * np.sin(X[:, 0])
    )

    return Y

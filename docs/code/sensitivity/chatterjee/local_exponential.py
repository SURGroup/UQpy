"""

Auxiliary file
==============================================

"""

import numpy as np


def evaluate(X: np.array) -> np.array:
    r"""A non-linear function that is used to demonstrate sensitivity index.

    .. math::
        f(x) = \exp(x_1 + 2*x_2)
    """

    Y = np.exp(X[:, 0] + 2 * X[:, 1])

    return Y

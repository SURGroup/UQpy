import numpy as np


def evaluate(X: np.array) -> np.array:
    r"""A non-linear function that is used to test Cramer-von Mises sensitivity index.

    .. math::
        f(x) = \exp(x_1 + 2*x_2)

    Parameters
    ----------
    X : np.array
        An `N*D` array holding values for each parameter, where `N` is the
        number of samples and `D` is the number of parameters
        (in this case, 2).

    Returns
    -------
    np.array
        [description]
    """

    Y = np.exp(X[:, 0] + 2 * X[:, 1])

    return Y

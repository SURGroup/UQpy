import numpy as np


class LeastSquareRegression:
    """
    Class to calculate the polynomial_chaos coefficients via the least-squares solution to
    the linear matrix equation. The equation may be under-, well-, or
    over-determined.

    **Inputs:**

    * **poly_object** ('class'):
        Object from the 'Polynomial' class

    **Methods:**

    """

    def __init__(self, polynomials, verbose=False):
        self.polynomials = polynomials
        self.verbose = verbose

    def run(self, x, y):
        """
        Least squares solution to compute the polynomial_chaos coefficients.

        **Inputs:**

        * **x** (`ndarray`):
            `ndarray` containing the training points (samples).

        * **y** (`ndarray`):
            `ndarray` containing the model evaluations (labels) at the
            training points.

        **Outputs:**

        * **c_** (`ndarray`):
            Returns the polynomial_chaos coefficients.

        """
        a = self.polynomials.evaluate(x)
        c_, res, rank, sing = np.linalg.lstsq(a, y)
        if c_.ndim == 1:
            c_ = c_.reshape(-1, 1)

        return c_

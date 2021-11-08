import numpy as np
from UQpy.surrogates.polynomial_chaos_new.polynomials.baseclass import Polynomials
from UQpy.surrogates.polynomial_chaos_new.regressions.baseclass.Regression import Regression


class LeastSquareRegression(Regression):

    def __init__(self, polynomial_basis: Polynomials):
        """
        Class to calculate the polynomial_chaos coefficients via the least-squares solution to
        the linear matrix equation. The equation may be under-, well-, or over-determined.


        :param polynomial_basis: Object from the 'Polynomial' class
        """
        super().__init__(polynomial_basis)

    def run(self, x, y):
        """
        Least squares solution to compute the polynomial_chaos coefficients.

        :param x: `ndarray` containing the training points (samples).
        :param y: `ndarray` containing the model evaluations (labels) at the training points.
        :return: Returns the polynomial_chaos coefficients.
        """
        a = self.polynomial_basis.evaluate_basis(x)
        c_, res, rank, sing = np.linalg.lstsq(a, y)
        if c_.ndim == 1:
            c_ = c_.reshape(-1, 1)

        return c_, None

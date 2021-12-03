import numpy as np

from UQpy.surrogates.polynomial_chaos.polynomials import PolynomialBasis
from UQpy.surrogates.polynomial_chaos.regressions.baseclass.Regression import Regression


class LeastSquareRegression(Regression):

    def run(self, x, y, design_matrix):
        """
        Least squares solution to compute the polynomial_chaos coefficients.

        :param x: `ndarray` containing the training points (samples).
        :param y: `ndarray` containing the model evaluations (labels) at the training points.
        :param design_matrix: matrix containing the evaluation of the polynomials at the input points x.
        :return: Returns the polynomial_chaos coefficients.
        """
        c_, res, rank, sing = np.linalg.lstsq(design_matrix, y)
        if c_.ndim == 1:
            c_ = c_.reshape(-1, 1)

        return c_, None, np.shape(c_)[1]

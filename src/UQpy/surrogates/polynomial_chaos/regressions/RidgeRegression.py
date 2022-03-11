import logging
import numpy as np

from UQpy.surrogates.polynomial_chaos.polynomials import PolynomialBasis
from UQpy.surrogates.polynomial_chaos.regressions.baseclass.Regression import Regression


class RidgeRegression(Regression):

    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000,
                 penalty: float = 1):
        """
        Class to calculate the polynomial_chaos coefficients with the Ridge regression method.


        :param learning_rate: Size of steps for the gradient descent.
        :param iterations: Number of iterations of the optimization algorithm.
        :param penalty: Penalty parameter controls the strength of regularization. When it
         is close to zero, then the ridge regression converges to the linear
         regression, while when it goes to infinity, polynomial_chaos coefficients
         converge to zero.
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.penalty = penalty
        self.logger = logging.getLogger(__name__)

    def run(self, x: np.ndarray, y: np.ndarray, design_matrix: np.ndarray):
        """
        Implements the LASSO method to compute the polynomial_chaos coefficients.

        :param x: :class:`numpy.ndarray` containing the training points (samples).
        :param y: :class:`numpy.ndarray` containing the model evaluations (labels) at the training points.
        :param design_matrix: matrix containing the evaluation of the polynomials at the input points **x**.
        :return: Weights (polynomial_chaos coefficients)  and Bias of the regressor
        """
        m, n = design_matrix.shape

        if y.ndim == 1 or y.shape[1] == 1:
            y = y.reshape(-1, 1)
            w = np.zeros(n).reshape(-1, 1)
            b = 0

            for _ in range(self.iterations):
                y_pred = (design_matrix.dot(w) + b).reshape(-1, 1)

                dw = (-(2 * design_matrix.T.dot(y - y_pred)) + (2 * self.penalty * w)) / m
                db = -2 * np.sum(y - y_pred) / m

                w = w - self.learning_rate * dw
                b = b - self.learning_rate * db

        else:
            n_out_dim = y.shape[1]
            w = np.zeros((n, n_out_dim))
            b = np.zeros(n_out_dim).reshape(1, -1)

            for _ in range(self.iterations):
                y_pred = design_matrix.dot(w) + b

                dw = (-(2 * design_matrix.T.dot(y - y_pred)) + (2 * self.penalty * w)) / m
                db = -2 * np.sum((y - y_pred), axis=0).reshape(1, -1) / m

                w = w - self.learning_rate * dw
                b = b - self.learning_rate * db

        return w, b, np.shape(w)[1]

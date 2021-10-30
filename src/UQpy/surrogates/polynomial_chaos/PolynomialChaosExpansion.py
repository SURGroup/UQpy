import logging

from beartype import beartype

from UQpy.surrogates.polynomial_chaos.regressions.LeastSquares import (
    LeastSquareRegression,
)
from UQpy.surrogates.polynomial_chaos.regressions.Ridge import RidgeRegression
from UQpy.surrogates.polynomial_chaos.regressions.Lasso import LassoRegression
from UQpy.surrogates.polynomial_chaos.regressions.baseclass.Regression import Regression


class PolynomialChaosExpansion:

    @beartype
    def __init__(self, regression_method: Regression):
        """
        Constructs a surrogate model based on the Polynomial Chaos Expansion (polynomial_chaos) method.

        :param regression_method: object for the method used for the calculation of the polynomial_chaos coefficients.
        """
        self.regression_method = regression_method
        self.logger = logging.getLogger(__name__)
        self.C = None
        self.b = None

    def fit(self, x, y):
        """
        Fit the surrogate model using the training samples and the corresponding model values. This method calls the
        'run' method of the input method class.

        :param x: `ndarray` containing the training points.
        :param y: `ndarray` containing the model evaluations at the training points.

        The ``fit`` method has no returns and it creates an `ndarray` with the
        polynomial_chaos coefficients.
        """
        self.logger.info("UQpy: Running polynomial_chaos.fit")

        if type(self.regression_method) == LeastSquareRegression:
            self.C = self.regression_method.run(x, y)

        elif (
            type(self.regression_method) == LassoRegression
            or type(self.regression_method) == RidgeRegression
        ):
            self.C, self.b = self.regression_method.run(x, y)

        self.logger.info("UQpy: polynomial_chaos fit complete.")

    def predict(self, points):
        """
        Predict the model response at new points.
        This method evaluates the polynomial_chaos model at new sample points.

        :param points: Points at which to predict the model response.
        :return: Predicted values at the new points.
        """
        a = self.regression_method.polynomials.evaluate(points)

        if type(self.regression_method) == LeastSquareRegression:
            y = a.dot(self.C)

        elif (
            type(self.regression_method) == LassoRegression
            or type(self.regression_method) == RidgeRegression
        ):
            y = a.dot(self.C) + self.b

        return y

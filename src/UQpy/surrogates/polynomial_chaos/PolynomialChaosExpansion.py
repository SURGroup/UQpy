import logging

from UQpy.surrogates.polynomial_chaos.regressions.LeastSquares import LeastSquareRegression
from UQpy.surrogates.polynomial_chaos.regressions.Ridge import RidgeRegression
from UQpy.surrogates.polynomial_chaos.regressions.Lasso import LassoRegression


class PolynomialChaosExpansion:
    """
    Constructs a surrogate model based on the Polynomial Chaos Expansion (polynomial_chaos)
    method.

    **Inputs:**

    * **method** (class):
        object for the method used for the calculation of the polynomial_chaos coefficients.

    **Methods:**

    """

    def __init__(self, method):
        self.method = method
        self.logger = logging.getLogger(__name__)
        self.C = None
        self.b = None

    def fit(self, x, y):
        """
        Fit the surrogate model using the training samples and the
        corresponding model values. This method calls the 'run' method of the
        input method class.

        **Inputs:**

        * **x** (`ndarray`):
            `ndarray` containing the training points.

        * **y** (`ndarray`):
            `ndarray` containing the model evaluations at the training points.

        **Output/Return:**

        The ``fit`` method has no returns and it creates an `ndarray` with the
        polynomial_chaos coefficients.
        """

        self.logger.info('UQpy: Running polynomial_chaos.fit')

        if type(self.method) == LeastSquareRegression:
            self.C = self.method.run(x, y)

        elif type(self.method) == LassoRegression or \
                type(self.method) == RidgeRegression:
            self.C, self.b = self.method.run(x, y)

        self.logger.info('UQpy: polynomial_chaos fit complete.')

    def predict(self, points):

        """
        Predict the model response at new points.
        This method evaluates the polynomial_chaos model at new sample points.

        **Inputs:**

        * **x_test** (`ndarray`):
            Points at which to predict the model response.

        **Outputs:**

        * **y** (`ndarray`):
            Predicted values at the new points.

        """

        a = self.method.polynomials.evaluate(points)

        if type(self.method) == LeastSquareRegression:
            y = a.dot(self.C)

        elif type(self.method) == LassoRegression or \
                type(self.method) == RidgeRegression:
            y = a.dot(self.C) + self.b

        return y

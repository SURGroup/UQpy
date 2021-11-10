import logging

import numpy as np
from beartype import beartype

from UQpy.surrogates.baseclass.Surrogate import Surrogate
from UQpy.surrogates.polynomial_chaos.regressions.baseclass.Regression import Regression


class PolynomialChaosExpansion(Surrogate):

    @beartype
    def __init__(self, regression_method: Regression):
        """
        Constructs a surrogate model based on the Polynomial Chaos Expansion (polynomial_chaos) method.

        :param regression_method: object for the method used for the calculation of the polynomial_chaos coefficients.
        """
        self.regression_method = regression_method
        self.logger = logging.getLogger(__name__)
        self.coefficients = None
        self.bias = None

    def fit(self, x, y):
        """
        Fit the surrogate model using the training samples and the corresponding model values. This method calls the
        'run' method of the input method class.

        :param x: containing the training points.
        :param y: containing the model evaluations at the training points.

        The :meth:`fit` method has no returns and it creates an `ndarray` with the
        polynomial_chaos coefficients.
        """
        self.logger.info("UQpy: Running polynomial_chaos.fit")
        self.coefficients, self.bias = self.regression_method.run(x, y)
        self.logger.info("UQpy: polynomial_chaos fit complete.")

    def predict(self, points, **kwargs):
        """
        Predict the model response at new points.
        This method evaluates the polynomial_chaos model at new sample points.

        :param points: Points at which to predict the model response.
        :return: Predicted values at the new points.
        """
        a = self.regression_method.polynomial_basis.evaluate_basis(points)
        y = a.dot(self.coefficients)
        if self.bias is not None:
            y = y + self.bias
        return y

    def validation_error(self, x, y):
        """
        Returns the validation error.

        :param x: `ndarray` containing the samples of the validation dataset.
        :param y: `ndarray` containing model evaluations for the validation dataset.
        :return: Validation error.
        """

        if y.ndim == 1 or y.shape[1] == 1:
            y = y.reshape(-1, 1)

        y_val = self.predict(x, )

        n_samples = x.shape[0]
        mu_yval = (1 / n_samples) * np.sum(y, axis=0)
        eps_val = ((n_samples - 1) / n_samples
                   * ((np.sum((y - y_val) ** 2, axis=0))
                      / (np.sum((y - mu_yval) ** 2, axis=0))))

        if y.ndim == 1 or y.shape[1] == 1:
            eps_val = float(eps_val)

        return np.round(eps_val, 7)

    def get_moments(self):
        """
        Returns the first two moments of the polynomial_chaos surrogate which are directly
        estimated from the polynomial_chaos coefficients.

        :return: Returns the mean and variance.
        """
        if self.bias is not None:
            mean = self.coefficients[0, :] + np.squeeze(self.bias)
        else:
            mean = self.coefficients[0, :]

        variance = np.sum(self.coefficients[1:] ** 2, axis=0)

        if self.coefficients.ndim == 1 or self.coefficients.shape[1] == 1:
            variance = float(variance)
            mean = float(mean)

        return np.round(mean, 4), np.round(variance, 4)

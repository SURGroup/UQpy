import logging
from typing import Union
import numpy as np

from UQpy.distributions.baseclass import Distribution
from UQpy.utilities.ValidationTypes import NumpyFloatArray
from UQpy.distributions import DistributionContinuous1D


class SROM:
    def __init__(
        self,
        samples: Union[list, np.ndarray],
        target_distributions: list[Distribution],
        moments:list = None,
        weights_errors: list = None,
        weights_distribution: Union[list, np.ndarray] = None,
        weights_moments: list = None,
        weights_correlation: np.ndarray = None,
        properties: list = None,
        correlation: np.ndarray = None,
    ):
        """
        Stochastic Reduced Order Model(stochastic_reduced_order_models) provide a low-dimensional, discrete
        approximation of a given random quantity.

        :param samples: An array/list of samples corresponding to the points at which the
         stochastic_reduced_order_models is defined.
        :param target_distributions: A list of distribution objects for each random variable.
        :param moments: A list containing first and second order moment about origin of all random variables.
        :param weights_errors: A list of weights associated with the error in distribution, moments and correlation.
         This corresponds to a list of the values :math:`a_{u}` in the objective function above.
         Default: weights_errors = :math:`[1, 0.2, 0]`
        :param weights_distribution: A list or array containing weights associated with matching the distribution at
         each sample value.
         `weights_distribution` is an array or list of shape :code:`(m, d)` where each weight corresponds to the weight
         :math:`w_F(x_{k,i}; i)` assigned for matching the distribution of component :code:`i` at sample point
         :math:`x_{k,i}`.
         If `weights_distribution` is :code:`(1, d)`, it is assumed that each sample is equally weighted according
         to the corresponding weight for its distribution.
         Default: An array of shape :code:`(m, d)` with all elements equal to :math:`1`.
        :param weights_moments: An list or array containing weights associated with matching the moments about the
         origin for each component.
         `weights_moments` is a list or array of shape `(2, d), where each weight corresponds to the weight
         :math:`w_{\mu}(r; i)` assigned for matching the moment of order :math:`r = 1, 2` for component :code:`i`.
         If `weights_moments` is :code:`(1, d)`, it is assumed that moments of all order are equally weighted.
         Default: :code:`weights_moments = [[1/(moment[0][i]^2)], [1/(moment[1][i]^2)]] for i = 1, 2, ..., d`.
        :param weights_correlation: A list or array containing weights associated with matching the correlation of the
         random variables. `weights_correlation` is a list or array of shape :code:`(d, d)` where each weight
         corresponds to the weight :math:`w_R(i, j)` assigned for matching the correlation between component :code:`i`
         and component :code:`j`
         Default: :code:`weights_correlation = (d, d)` array with all elements equal to :math:`1`.
        :param properties: A list of booleans declaring the properties to be matched in the reduced order model.

         - :code:`properties[0] = True` matches the marginal distributions

         - :code:`properties[1] = True` matches the mean values

         - :code:`properties[2] = True` matches the mean square

         - :code:`properties[3] = True` matches the correlation

        :param correlation: Correlation matrix between random variables.

        """
        self.target_distributions = target_distributions
        self.correlation = correlation
        self.moments = moments

        self.weights_distribution = weights_distribution
        self.weights_moments = weights_moments
        self.weights_correlation = weights_correlation
        self.weights_errors = weights_errors

        self.properties = properties
        self.logger = logging.getLogger(__name__)
        self.sample_weights: NumpyFloatArray = None
        """The probability weights defining discrete approximation of continuous random variables."""

        if isinstance(samples, list):
            self.samples = np.array(samples)
            self.samples_number = self.samples.shape[0]
            self.dimension = self.samples.shape[1]
        elif isinstance(samples, np.ndarray):
            self.dimension = samples.shape[1]
            self.samples = samples
            self.samples_number = samples.shape[0]
        else:
            raise NotImplementedError("UQpy: 'samples' should be a list or numpy array")

        if self.target_distributions is None:
            raise NotImplementedError("UQpy: Target Distribution is not defined.")

        if isinstance(self.target_distributions, list):
            for i in range(len(self.target_distributions)):
                if not isinstance(
                    self.target_distributions[i], DistributionContinuous1D
                ):
                    raise TypeError(
                        "UQpy: A DistributionContinuous1D object must be provided."
                    )

        if self.properties is not None:
            self.run()
        else:
            self.logger.info(
                "UQpy: No properties list provided, execute the stochastic_reduced_order_models by calling"
                " run method and specifying a properties list"
            )

    def run(
        self,
        weights_errors: list = None,
        weights_distribution: list = None,
        weights_moments: list = None,
        weights_correlation: list = None,
        properties: list = None,
    ):
        """
        Execute the stochastic reduced order model in the :class:`.SROM` class.

        The :meth:`run` method is the function that computes the probability weights corresponding to the sample. If
        `properties` is provided, the :meth:`run` method is automatically called when the :class:`.SROM`
        object is defined. The user may also call the :meth:`run` method directly to generate samples. The :meth:`run`
        method of the :class:`.SROM` class can be invoked many times with different weights parameters and
        each time computed probability weights are overwritten.

        :param weights_errors: A list of weights associated with the error in distribution, moments and correlation.
         This corresponds to a list of the values :math:`a_{u}` in the objective function above.
         Default: weights_errors = :math:`[1, 0.2, 0]`
        :param weights_distribution: A list or array containing weights associated with matching the distribution at
         each sample value.
         `weights_distribution` is an array or list of shape :code:`(m, d)` where each weight corresponds to the weight
         :math:`w_F(x_{k,i}; i)` assigned for matching the distribution of component :code:`i` at sample point
         :math:`x_{k,i}`.
         If `weights_distribution` is :code:`(1, d)`, it is assumed that each sample is equally weighted according
         to the corresponding weight for its distribution.
         Default: An array of shape :code:`(m, d)` with all elements equal to :math:`1`.
        :param weights_moments: An list or array containing weights associated with matching the moments about the
         origin for each component.
         `weights_moments` is a list or array of shape :code:`(2, d)`, where each weight corresponds to the weight
         :math:`w_{\mu}(r; i)` assigned for matching the moment of order :math:`r = 1, 2` for component :code:`i`.
         If `weights_moments` is :code:`(1, d)`, it is assumed that moments of all order are equally weighted.
         Default: :code:`weights_moments = [[1/(moment[0][i]^2)], [1/(moment[1][i]^2)]] for i = 1, 2, ..., d`.
        :param weights_correlation: A list or array containing weights associated with matching the correlation of the
         random variables.
         `weights_correlation` is a list or array of shape :code:`(d, d)` where each weight corresponds to the weight
         :math:`w_R(i, j)` assigned for matching the correlation between component :code:`i` and component :code:`j`
         Default: :code:`weights_correlation = (d, d)` array with all elements equal to :math:`1`.
        :param properties: A list of booleans declaring the properties to be matched in the reduced order model.

         - :code:`properties[0] = True` matches the marginal distributions
         - :code:`properties[1] = True` matches the mean values
         - :code:`properties[2] = True` matches the mean square
         - :code:`properties[3] = True` matches the correlation
        """
        from scipy import optimize

        self.weights_distribution = weights_distribution
        self.weights_moments = weights_moments
        self.weights_correlation = weights_correlation
        self.weights_errors = weights_errors
        self.properties = properties

        # Check properties to match
        if self.properties is None:
            self.properties = [True, True, True, False]

        self._init_srom()

        self.logger.info("UQpy: Performing stochastic_reduced_order_models...")

        def f(p0, samples, wd, wm, wc, mar, n, d, m, alpha, prop, correlation):
            e1 = 0.0
            e2 = 0.0
            e22 = 0.0
            e3 = 0.0
            com = np.append(samples, np.atleast_2d(p0).T, 1)
            for j in range(d):
                srt = com[np.argsort(com[:, j].flatten())]
                s = srt[:, j]
                a = srt[:, d]
                a0 = np.cumsum(a)
                marginal = mar[j].cdf

                if prop[0] is True:
                    for i in range(n):
                        e1 += wd[i, j] * (a0[i] - marginal(s[i])) ** 2

                if prop[1] is True:
                    e2 += wm[0, j] * (np.sum(p0 * samples[:, j]) - m[0, j]) ** 2

                if prop[2] is True:
                    e22 += (
                        wm[1, j]
                        * (
                            np.sum(np.array(p0) * (samples[:, j] * samples[:, j]))
                            - m[1, j]
                        )
                        ** 2
                    )

                if prop[3] is True:
                    for k in range(d):
                        if k > j:
                            r = (
                                correlation[j, k]
                                * np.sqrt(
                                    (m[1, j] - m[0, j] ** 2) * (m[1, k] - m[0, k] ** 2)
                                )
                                + m[0, j] * m[0, k]
                            )
                            e3 += (
                                wc[k, j]
                                * (np.sum(p0 * (samples[:, j] * samples[:, k])) - r)
                                ** 2
                            )

            return alpha[0] * e1 + alpha[1] * (e2 + e22) + alpha[2] * e3

        def constraint(x):
            return np.sum(x) - 1

        cons = {"type": "eq", "fun": constraint}

        p_ = optimize.minimize(
            f,
            np.zeros(self.samples_number),
            args=(
                self.samples,
                self.weights_distribution,
                self.weights_moments,
                self.weights_correlation,
                self.target_distributions,
                self.samples_number,
                self.dimension,
                self.moments,
                self.weights_errors,
                self.properties,
                self.correlation,
            ),
            constraints=cons,
            method="SLSQP",
            bounds=[[0, 1]] * self.samples_number,
        )

        self.sample_weights = p_.x
        """The probability weights defining discrete approximation of continuous random variables."""
        self.logger.info("UQpy: stochastic_reduced_order_models completed!")

    def _init_srom(self):
        if isinstance(self.moments, list):
            self.moments = np.array(self.moments)

        if isinstance(self.correlation, list):
            self.correlation = np.array(self.correlation)

        # Check moments and correlation
        if (
            self.properties[1] is True
            or self.properties[2] is True
            or self.properties[3] is True
        ):
            if self.moments is None:
                raise NotImplementedError("UQpy: 'moments' are required")
        # Both moments are required, if correlation property is required to be match
        if self.properties[3] is True:
            if self.moments.shape != (2, self.dimension):
                raise NotImplementedError("UQpy: Shape of 'moments' is not correct")
            if self.correlation is None:
                self.correlation = np.identity(self.dimension)
        # moments.shape[0] should be 1 or 2
        if self.moments.shape != (1, self.dimension) and self.moments.shape != (
            2,
            self.dimension,
        ):
            raise NotImplementedError("UQpy: Shape of 'moments' is not correct")
        # If both the moments are to be included in objective function, then moments.shape[0] should be 2
        if self.properties[1] is True and self.properties[2] is True:
            if self.moments.shape != (2, self.dimension):
                raise NotImplementedError("UQpy: Shape of 'moments' is not correct")
        # If only second order moment is to be included in objective function and moments.shape[0] is 1. Then
        # self.moments is converted shape = (2, self.dimension) where is second row contain second order moments.
        if self.properties[1] is False and self.properties[2] is True:
            if self.moments.shape == (1, self.dimension):
                temp = np.ones(shape=(1, self.dimension))
                self.moments = np.concatenate((temp, self.moments))

        # Check weights corresponding to errors
        if self.weights_errors is None:
            self.weights_errors = [1, 0.2, 0]
        elif isinstance(self.weights_errors, list):
            self.weights_errors = np.array(self.weights_errors).astype(np.float64)
        elif not isinstance(self.weights_errors, np.ndarray):
            raise NotImplementedError(
                "UQpy: weights_errors attribute should be a list or numpy array"
            )

        # Check weights corresponding to distribution
        if self.weights_distribution is None:
            self.weights_distribution = np.ones(
                shape=(self.samples.shape[0], self.dimension)
            )
        elif isinstance(self.weights_distribution, list):
            self.weights_distribution = np.array(self.weights_distribution)
        elif not isinstance(self.weights_distribution, np.ndarray):
            raise NotImplementedError(
                "UQpy: weights_distribution attribute should be a list or numpy array"
            )

        if self.weights_distribution.shape == (1, self.dimension):
            self.weights_distribution = self.weights_distribution * np.ones(
                shape=(self.samples.shape[0], self.dimension)
            )
        elif self.weights_distribution.shape != (self.samples.shape[0], self.dimension):
            raise NotImplementedError(
                "UQpy: Size of 'weights for distribution' is not correct"
            )

        # Check weights corresponding to moments and it's default list
        if self.weights_moments is None:
            self.weights_moments = np.reciprocal(np.square(self.moments))
        elif isinstance(self.weights_moments, list):
            self.weights_moments = np.array(self.weights_moments)
        elif not isinstance(self.weights_moments, np.ndarray):
            raise NotImplementedError(
                "UQpy: weights_moments attribute should be a list or numpy array"
            )

        if self.weights_moments.shape == (1, self.dimension):
            self.weights_moments = self.weights_moments * np.ones(
                shape=(2, self.dimension)
            )
        elif self.weights_moments.shape != (2, self.dimension):
            raise NotImplementedError(
                "UQpy: Size of 'weights for moments' is not correct"
            )

        # Check weights corresponding to correlation and it's default list
        if self.weights_correlation is None:
            self.weights_correlation = np.ones(shape=(self.dimension, self.dimension))
        elif isinstance(self.weights_correlation, list):
            self.weights_correlation = np.array(self.weights_correlation)
        elif not isinstance(self.weights_correlation, np.ndarray):
            raise NotImplementedError(
                "UQpy: weights_correlation attribute should be a list or numpy array"
            )

        if self.weights_correlation.shape != (self.dimension, self.dimension):
            raise NotImplementedError(
                "UQpy: Size of 'weights for correlation' is not correct"
            )

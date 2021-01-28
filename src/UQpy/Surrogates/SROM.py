import numpy as np
from UQpy.Distributions import DistributionContinuous1D

########################################################################################################################
########################################################################################################################
#                                         Stochastic Reduced Order Model  (SROM)                                       #
########################################################################################################################
########################################################################################################################


class SROM:

    """
    Stochastic Reduced Order Model(SROM) provide a low-dimensional, discrete approximation of a given random
    quantity.

    **Inputs:**

    * **samples** (`ndarray`):
        An array/list of samples corresponding to the points at which the SROM is defined.

    * **target_dist_object** ((list of) ``Distribution`` object(s)):
        A list of distribution objects for each random variable.

    * **moments** (`list` of `float`):
        A list containing first and second order moment about origin of all random variables.

    * **weights_errors** (`list` of `float`):
        A list of weights associated with the error in distribution, moments and correlation.

        This corresponds to a list of the values :math:`a_{u}` in the objective function above.

        Default: weights_errors = [1, 0.2, 0]

    * **properties** (`list` of `booleans`):
        A list of booleans declaring the properties to be matched in the reduced order model.

        `properties[0] = True` matches the marginal distributions

        `properties[1] = True` matches the mean values

        `properties[2] = True` matches the mean square

        `properties[3] = True` matches the correlation

    * **weights_distribution** (`ndarray` or `list` of `float`):
        A list or array containing weights associated with matching the distribution at each sample value.

        `weights_distribution` is an array or list of shape `(m, d)` where each weight corresponds to the weight
        :math:`w_F(x_{k,i}; i)` assigned for matching the distribution of component `i` at sample point
        :math:`x_{k,i}`.

        If `weights_distribution` is `(1, d)`, it is assumed that each sample sample is equally weighted according
        to the corresponding weight for its distribution.

        Default: `weights_distribution` = An array of shape `(m, d)` with all elements equal to 1.

    * **weights_moments** (`ndarray` or `list` of `float`):
        An list or array containing weights associated with matching the moments about the origin for each
        component.

        `weights_moments` is a list or array of shape `(2, d), where each weight corresponds to the weight
        :math:`w_{\mu}(r; i)` assigned for matching the moment of order :math:`r = 1, 2` for component `i`.

        If `weights_moments` is `(1, d)`, it is assumed that moments of all order are equally weighted.

        Default: `weights_moments` = [[1/(moment[0][i]^2)], [1/(moment[1][i]^2)]] for i = 1, 2, ..., d.

    * **weights_correlation** (`ndarray` or `list` of `float`):
        A list or array containing weights associated with matching the correlation of the random variables.

        `weights_correlation` is a list or array of shape `(d, d)` where each weight corresponds to the weight
        :math:`w_R(i, j)` assigned for matching the correlation between component `i` and component `j`

        Default: `weights_correlation` = `(d, d)` array with all elements equal to 1.

    * **correlation** (`ndarray` or `list of floats`):
        Correlation matrix between random variables.

    **Attribute:**

    * **sample_weights** (`ndarray`):
        The probability weights defining discrete approximation of continuous random variables.

    **Methods:**

    """

    def __init__(self, samples, target_dist_object, moments=None, weights_errors=None, weights_distribution=None,
                 weights_moments=None, weights_correlation=None, properties=None, correlation=None, verbose=False):

        self.target_dist_object = target_dist_object
        self.correlation = correlation
        self.moments = moments

        self.weights_distribution = weights_distribution
        self.weights_moments = weights_moments
        self.weights_correlation = weights_correlation
        self.weights_errors = weights_errors

        self.properties = properties
        self.verbose = verbose
        self.sample_weights = None

        if isinstance(samples, list):
            self.samples = np.array(samples)
            self.nsamples = self.samples.shape[0]
            self.dimension = self.samples.shape[1]
        elif isinstance(samples, np.ndarray):
            self.dimension = samples.shape[1]
            self.samples = samples
            self.nsamples = samples.shape[0]
        else:
            raise NotImplementedError("UQpy: 'samples' sholud be a list or numpy array")

        if self.target_dist_object is None:
            raise NotImplementedError("UQpy: Target Distribution is not defined.")

        if isinstance(self.target_dist_object, list):
            for i in range(len(self.target_dist_object)):
                if not isinstance(self.target_dist_object[i], DistributionContinuous1D):
                    raise TypeError('UQpy: A DistributionContinuous1D object must be provided.')

        if self.properties is not None:
            self.run()
        else:
            print('UQpy: No properties list provided, execute the SROM by calling run method and specifying a '
                  'properties list')

    def run(self, weights_errors=None, weights_distribution=None, weights_moments=None, weights_correlation=None,
            properties=None):
        """
        Execute the stochastic reduced order model in the ``SROM`` class.

        The ``run`` method is the function that computes the probability weights corresponding to the sample. If
        `properties` is provided, the ``run`` method is automatically called when the ``SROM`` object is defined. The
        user may also call the ``run`` method directly to generate samples. The ``run`` method of the ``SROM`` class can
        be invoked many times with different weights parameters and each time computed probability weights are
        overwritten.

        **Inputs:**

        * **weights_errors** (`list` of `float`):
            A list of weights associated with the error in distribution, moments and correlation.

            This corresponds to a list of the values :math:`a_{u}` in the objective function above.

            Default: weights_errors = [1, 0.2, 0]

        * **properties** (`list` of `booleans`):
            A list of booleans declaring the properties to be matched in the reduced order model.

            `properties[0] = True` matches the marginal distributions

            `properties[1] = True` matches the mean values

            `properties[2] = True` matches the mean square

            `properties[3] = True` matches the correlation

        * **weights_distribution** (`ndarray` or `list` of `float`):
            A list or array containing weights associated with matching the distribution at each sample value.

            `weights_distribution` is an array or list of shape `(m, d)` where each weight corresponds to the weight
            :math:`w_F(x_{k,i}; i)` assigned for matching the distribution of component `i` at sample point
            :math:`x_{k,i}`.

            If `weights_distribution` is `(1, d)`, it is assumed that each sample sample is equally weighted according
            to the corresponding weight for its distribution.

            Default: `weights_distribution` = An array of shape `(m, d)` with all elements equal to 1.

        * **weights_moments** (`ndarray` or `list` of `float`):
            An list or array containing weights associated with matching the moments about the origin for each
            component.

            `weights_moments` is a list or array of shape `(2, d), where each weight corresponds to the weight
            :math:`w_{\mu}(r; i)` assigned for matching the moment of order :math:`r = 1, 2` for component `i`.

            If `weights_moments` is `(1, d)`, it is assumed that moments of all order are equally weighted.

            Default: `weights_moments` = [[1/(moment[0][i]^2)], [1/(moment[1][i]^2)]] for i = 1, 2, ..., d.

        * **weights_correlation** (`ndarray` or `list` of `float`):
            A list or array containing weights associated with matching the correlation of the random variables.

            `weights_correlation` is a list or array of shape `(d, d)` where each weight corresponds to the weight
            :math:`w_R(i, j)` assigned for matching the correlation between component `i` and component `j`

            Default: `weights_correlation` = `(d, d)` array with all elements equal to 1.

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

        if self.verbose:
            print('UQpy: Performing SROM...')

        def f(p0, samples, wd, wm, wc, mar, n, d, m, alpha, prop, correlation):
            e1 = 0.
            e2 = 0.
            e22 = 0.
            e3 = 0.
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
                    e22 += wm[1, j] * (
                            np.sum(np.array(p0) * (samples[:, j] * samples[:, j])) - m[1, j]) ** 2

                if prop[3] is True:
                    for k in range(d):
                        if k > j:
                            r = correlation[j, k] * np.sqrt((m[1, j] - m[0, j] ** 2) * (m[1, k] - m[0, k] ** 2)) + \
                                m[0, j] * m[0, k]
                            e3 += wc[k, j] * (np.sum(p0 * (samples[:, j] * samples[:, k])) - r) ** 2

            return alpha[0] * e1 + alpha[1] * (e2 + e22) + alpha[2] * e3

        def constraint(x):
            return np.sum(x) - 1

        cons = {'type': 'eq', 'fun': constraint}

        p_ = optimize.minimize(f, np.zeros(self.nsamples),
                               args=(self.samples, self.weights_distribution, self.weights_moments,
                                     self.weights_correlation, self.target_dist_object, self.nsamples, self.dimension,
                                     self.moments, self.weights_errors, self.properties, self.correlation),
                               constraints=cons, method='SLSQP', bounds=[[0, 1]]*self.nsamples)

        self.sample_weights = p_.x
        if self.verbose:
            print('UQpy: SROM completed!')

    def _init_srom(self):
        """
        Initialization and preliminary error checks.
        """
        if isinstance(self.moments, list):
            self.moments = np.array(self.moments)

        if isinstance(self.correlation, list):
            self.correlation = np.array(self.correlation)

        # Check moments and correlation
        if self.properties[1] is True or self.properties[2] is True or self.properties[3] is True:
            if self.moments is None:
                raise NotImplementedError("UQpy: 'moments' are required")
        # Both moments are required, if correlation property is required to be match
        if self.properties[3] is True:
            if self.moments.shape != (2, self.dimension):
                raise NotImplementedError("UQpy: Shape of 'moments' is not correct")
            if self.correlation is None:
                self.correlation = np.identity(self.dimension)
        # moments.shape[0] should be 1 or 2
        if self.moments.shape != (1, self.dimension) and self.moments.shape != (2, self.dimension):
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
            raise NotImplementedError("UQpy: weights_errors attribute should be a list or numpy array")

        # Check weights corresponding to distribution
        if self.weights_distribution is None:
            self.weights_distribution = np.ones(shape=(self.samples.shape[0], self.dimension))
        elif isinstance(self.weights_distribution, list):
            self.weights_distribution = np.array(self.weights_distribution)
        elif not isinstance(self.weights_distribution, np.ndarray):
            raise NotImplementedError("UQpy: weights_distribution attribute should be a list or numpy array")

        if self.weights_distribution.shape == (1, self.dimension):
            self.weights_distribution = self.weights_distribution * np.ones(shape=(self.samples.shape[0],
                                                                                   self.dimension))
        elif self.weights_distribution.shape != (self.samples.shape[0], self.dimension):
            raise NotImplementedError("UQpy: Size of 'weights for distribution' is not correct")

        # Check weights corresponding to moments and it's default list
        if self.weights_moments is None:
            self.weights_moments = np.reciprocal(np.square(self.moments))
        elif isinstance(self.weights_moments, list):
            self.weights_moments = np.array(self.weights_moments)
        elif not isinstance(self.weights_moments, np.ndarray):
            raise NotImplementedError("UQpy: weights_moments attribute should be a list or numpy array")

        if self.weights_moments.shape == (1, self.dimension):
            self.weights_moments = self.weights_moments * np.ones(shape=(2, self.dimension))
        elif self.weights_moments.shape != (2, self.dimension):
            raise NotImplementedError("UQpy: Size of 'weights for moments' is not correct")

        # Check weights corresponding to correlation and it's default list
        if self.weights_correlation is None:
            self.weights_correlation = np.ones(shape=(self.dimension, self.dimension))
        elif isinstance(self.weights_correlation, list):
            self.weights_correlation = np.array(self.weights_correlation)
        elif not isinstance(self.weights_correlation, np.ndarray):
            raise NotImplementedError("UQpy: weights_correlation attribute should be a list or numpy array")

        if self.weights_correlation.shape != (self.dimension, self.dimension):
            raise NotImplementedError("UQpy: Size of 'weights for correlation' is not correct")
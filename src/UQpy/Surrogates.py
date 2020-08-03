# UQpy is distributed under the MIT license.
#
# Copyright (C) 2018  -- Michael D. Shields
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""This module contains functionality for all the surrogate methods supported in UQpy.

The module currently contains the following classes:

- ``SROM``: Class to estimate a discrete approximation for a continuous random variable using Stochastic Reduced Order
            Model.
- ``Kriging``: Class to generate an approximate surrogate model using Kriging.
"""

import numpy as np
import scipy.stats as stats

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


########################################################################################################################
########################################################################################################################
#                                         Kriging Interpolation  (Kriging)                                             #
########################################################################################################################
########################################################################################################################

class Kriging:
    """
    Kriging generates an Gaussian process regression-based surrogate model to predict the model output at new sample
    points.

    **Inputs:**

    * **reg_model** (`str` or `function`):
        `reg_model` specifies and evaluates the basis functions and their coefficients, which defines the trend of
        the model.

        Built-in options (string input): 'Constant', 'Linear', 'Quadratic'

        The user may also pass a callable function as defined in `User-Defined Regression Model` above.

    * **corr_model** (`str` or `function`):
        `corr_model` specifies and evaluates the correlation function.

        Built-in options (string input): 'Exponential', 'Gaussian', 'Linear', 'Spherical', 'Cubic', 'Spline'

        The user may also pass a callable function as defined in `User-Defined Correlation` above.

    * **corr_model_params** (`ndarray` or `list of floats`):
        List or array of initial values for the correlation model hyperparameters/scale parameters.

    * **bounds** (`list` of `float`):
        Bounds on the hyperparameters used to solve optimization problem to estimate maximum likelihood estimator.
        This should be a closed bound.

        Default: [0.001, 10**7] for each hyperparameter.

    * **op** (`boolean`):
        Indicator to solve MLE problem or not. If 'True' corr_model_params will be used as initial solution for
        optimization problem. Otherwise, corr_model_params will be directly use as the hyperparamters.

        Default: True.

    * **nopt** (`int`):
            Number of times MLE optimization problem is to be solved with a random starting point.

            Default: 1.

    * **verbose** (`Boolean`):
            A boolean declaring whether to write text to the terminal.

            Default value: False

    **Attributes:**

    * **beta** (`ndarray`):
            Regression coefficients.

    * **err_var** (`ndarray`):
            Variance of the Gaussian random process.

    * **C_inv** (`ndarray`):
            Inverse Cholesky decomposition of the correlation matrix.

    **Methods:**

    """

    def __init__(self, reg_model='Linear', corr_model='Exponential', bounds=None, op=True, nopt=1, normalize=True,
                 verbose=False, corr_model_params=None, optimizer=None, random_state=None, **kwargs_optimizer):

        self.reg_model = reg_model
        self.corr_model = corr_model
        self.corr_model_params = np.array(corr_model_params)
        self.bounds = bounds
        self.optimizer = optimizer
        self.nopt = nopt
        self.op = op
        self.normalize = normalize
        self.verbose = verbose
        self.random_state = random_state
        self.kwargs_optimizer = kwargs_optimizer

        # Variables are used outside the __init__
        self.samples = None
        self.values = None
        self.sample_mean, self.sample_std = None, None
        self.value_mean, self.value_std = None, None
        self.rmodel, self.cmodel = None, None
        self.beta, self.gamma, self.err_var = None, None, None
        self.F_dash, self.C_inv, self.G = None, None, None
        self.F, self.R = None, None

        # Initialize and run preliminary error checks.
        if self.reg_model is None:
            raise NotImplementedError("UQpy: Regression model is not defined.")

        if self.corr_model is None:
            raise NotImplementedError("Uqpy: Correlation model is not defined.")

        if self.corr_model_params is None:
            raise NotImplementedError("UQpy: corr_model_params is not defined.")

        if self.bounds is None:
            self.bounds = [[0.001, 10**7]]*self.corr_model_params.shape[0]

        if self.optimizer is None:
            from scipy.optimize import fmin_l_bfgs_b
            self.optimizer = fmin_l_bfgs_b
            self.kwargs_optimizer = {'bounds': self.bounds}
        elif callable(self.optimizer):
            self.optimizer = self.optimizer
        else:
            raise TypeError('UQpy: Input optimizer should be None (set to scipy.optimize.minimize) or a callable.')

        if type(self.reg_model).__name__ == 'function':
            self.rmodel = 'User defined'
            self.reg_model = self.reg_model
        elif self.reg_model in ['Constant', 'Linear', 'Quadratic']:
            self.rmodel = self.reg_model
            self.reg_model = self._regress()
        else:
            raise NotImplementedError("UQpy: Doesn't recognize the Regression model.")

        if type(self.corr_model).__name__ == 'function':
            self.cmodel = 'User defined'
            self.corr_model = self.corr_model
        elif self.corr_model in ['Exponential', 'Gaussian', 'Linear', 'Spherical', 'Cubic', 'Spline', 'Other']:
            self.cmodel = self.corr_model
            self.corr_model: callable = self._corr()
        else:
            raise NotImplementedError("UQpy: Doesn't recognize the Correlation model.")

        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

    def fit(self, samples, values, nopt=None, corr_model_params=None):
        """
        Fit the surrogate model using the training samples and the corresponding model values.

        The user can run this method multiple time after initiating the ``Kriging`` class object.

        This method updates the samples and parameters of the ``Kriging`` object. This method uses `corr_model_params`
        from previous run as the starting point for MLE problem unless user provides a new starting point.

        **Inputs:**

        * **samples** (`ndarray`):
            `ndarray` containing the training points.

        * **values** (`ndarray`):
            `ndarray` containing the model evaluations at the training points.

        **Output/Return:**

        The ``fit`` method has no returns, although it creates the `beta`, `err_var` and `C_inv` attributes of the
        ``Kriging`` class.

        """
        from scipy.linalg import cholesky

        if self.verbose:
            print('UQpy: Running Kriging.fit')

        def log_likelihood(p0, cm, s, f, y):
            # Return the log-likelihood function and it's gradient. Gradient is calculate using Central Difference
            m = s.shape[0]
            n = s.shape[1]
            r__, dr_ = cm(x=s, s=s, params=p0, dt=True)
            try:
                cc = cholesky(r__ + 2**(-52) * np.eye(m), lower=True)
            except np.linalg.LinAlgError:
                return np.inf, np.zeros(n)

            # Product of diagonal terms is negligible sometimes, even when cc exists.
            if np.prod(np.diagonal(cc)) == 0:
                return np.inf, np.zeros(n)

            cc_inv = np.linalg.inv(cc)
            r_inv = np.matmul(cc_inv.T, cc_inv)
            f__ = cc_inv.dot(f)
            y__ = cc_inv.dot(y)

            q__, g__ = np.linalg.qr(f__)  # Eq: 3.11, DACE

            # Check if F is a full rank matrix
            if np.linalg.matrix_rank(g__) != min(np.size(f__, 0), np.size(f__, 1)):
                raise NotImplementedError("Chosen regression functions are not sufficiently linearly independent")

            # Design parameters
            beta_ = np.linalg.solve(g__, np.matmul(np.transpose(q__), y__))

            # Computing the process variance (Eq: 3.13, DACE)
            sigma_ = np.zeros(y.shape[1])

            ll = 0
            for out_dim in range(y.shape[1]):
                sigma_[out_dim] = (1 / m) * (np.linalg.norm(y__[:, out_dim] - np.matmul(f__, beta_[:, out_dim])) ** 2)
                # Objective function:= log(det(sigma**2 * R)) + constant
                ll = ll + (np.log(np.linalg.det(sigma_[out_dim] * r__)) + m * (np.log(2 * np.pi) + 1))/2

            # Gradient of loglikelihood
            # Reference: C. E. Rasmussen & C. K. I. Williams, Gaussian Processes for Machine Learning, the MIT Press,
            # 2006, ISBN 026218253X. (Page 114, Eq.(5.9))
            residual = y - np.matmul(f, beta_)
            gamma = np.matmul(r_inv, residual)
            grad_mle = np.zeros(n)
            for in_dim in range(n):
                r_inv_derivative = np.matmul(r_inv, np.matmul(dr_[:, :, in_dim], r_inv))
                tmp = np.matmul(residual.T, np.matmul(r_inv_derivative, residual))
                for out_dim in range(y.shape[1]):
                    alpha = gamma / sigma_[out_dim]
                    tmp1 = np.matmul(alpha, alpha.T) - r_inv / sigma_[out_dim]
                    cov_der = sigma_[out_dim] * dr_[:, :, in_dim] + tmp * r__ / m
                    grad_mle[in_dim] = grad_mle[in_dim] - 0.5 * np.trace(np.matmul(tmp1, cov_der))

            return ll, grad_mle

        if nopt is not None:
            self.nopt = nopt
        if corr_model_params is not None:
            self.corr_model_params = corr_model_params
        self.samples = np.array(samples)

        # Number of samples and dimensions of samples and values
        nsamples, input_dim = self.samples.shape
        output_dim = int(np.size(values) / nsamples)

        self.values = np.array(values).reshape(nsamples, output_dim)

        # Normalizing the data
        if self.normalize:
            self.sample_mean, self.sample_std = np.mean(self.samples, 0), np.std(self.samples, 0)
            self.value_mean, self.value_std = np.mean(self.values, 0), np.std(self.values, 0)
            s_ = (self.samples - self.sample_mean)/self.sample_std
            y_ = (self.values - self.value_mean)/self.value_std
        else:
            s_ = self.samples
            y_ = self.values

        self.F, jf_ = self.reg_model(s_)

        # Maximum Likelihood Estimation : Solving optimization problem to calculate hyperparameters
        if self.op:
            starting_point = self.corr_model_params
            minimizer, fun_value = np.zeros([self.nopt, input_dim]), np.zeros([self.nopt, 1])
            for i__ in range(self.nopt):
                p_ = self.optimizer(log_likelihood, starting_point, args=(self.corr_model, s_, self.F, y_),
                                    **self.kwargs_optimizer)
                minimizer[i__, :] = p_[0]
                fun_value[i__, 0] = p_[1]
                # Generating new starting points using log-uniform distribution
                if i__ != self.nopt - 1:
                    starting_point = stats.reciprocal.rvs([j[0] for j in self.bounds], [j[1] for j in self.bounds], 1,
                                                          random_state=self.random_state)
            if min(fun_value) == np.inf:
                raise NotImplementedError("Maximum likelihood estimator failed: Choose different starting point or "
                                          "increase nopt")
            t = np.argmin(fun_value)
            self.corr_model_params = minimizer[t, :]

        # Updated Correlation matrix corresponding to MLE estimates of hyperparameters
        self.R = self.corr_model(x=s_, s=s_, params=self.corr_model_params)
        # Compute the regression coefficient (solving this linear equation: F * beta = Y)
        c = np.linalg.cholesky(self.R)                   # Eq: 3.8, DACE
        c_inv = np.linalg.inv(c)
        f_dash = np.linalg.solve(c, self.F)
        y_dash = np.linalg.solve(c, y_)
        q_, g_ = np.linalg.qr(f_dash)                 # Eq: 3.11, DACE
        # Check if F is a full rank matrix
        if np.linalg.matrix_rank(g_) != min(np.size(self.F, 0), np.size(self.F, 1)):
            raise NotImplementedError("Chosen regression functions are not sufficiently linearly independent")
        # Design parameters (beta: regression coefficient)
        self.beta = np.linalg.solve(g_, np.matmul(np.transpose(q_), y_dash))

        # Design parameter (R * gamma = Y - F * beta = residual)
        self.gamma = np.linalg.solve(c.T, (y_dash - np.matmul(f_dash, self.beta)))

        # Computing the process variance (Eq: 3.13, DACE)
        self.err_var = np.zeros(output_dim)
        for i in range(output_dim):
            self.err_var[i] = (1 / nsamples) * (np.linalg.norm(y_dash[:, i] - np.matmul(f_dash, self.beta[:, i])) ** 2)

        self.F_dash, self.C_inv, self.G = f_dash, c_inv, g_

        if self.verbose:
            print('UQpy: Kriging fit complete.')

    def predict(self, x, return_std=False):
        """
        Predict the model response at new points.

        This method evaluates the regression and correlation model at new sample points. Then, it predicts the function
        value and standard deviation.

        **Inputs:**

        * **x** (`list` or `numpy array`):
            Points at which to predict the model response.

        * **return_std** (`Boolean`):
            Indicator to estimate standard deviation.

        **Outputs:**

        * **f_x** (`numpy array`):
            Predicted values at the new points.

        * **std_f_x** (`numpy array`):
            Standard deviation of predicted values at the new points.

        """
        x_ = np.atleast_2d(x)
        if self.normalize:
            x_ = (x_ - self.sample_mean)/self.sample_std
            s_ = (self.samples - self.sample_mean) / self.sample_std
        else:
            s_ = self.samples
        fx, jf = self.reg_model(x_)
        rx = self.corr_model(x=x_, s=s_, params=self.corr_model_params)
        y = np.einsum('ij,jk->ik', fx, self.beta) + np.einsum('ij,jk->ik', rx, self.gamma)
        if self.normalize:
            y = self.value_mean + y * self.value_std
        if x_.shape[1] == 1:
            y = y.flatten()
        if return_std:
            r_dash = np.matmul(self.C_inv, rx.T)
            u = np.matmul(self.F_dash.T, r_dash) - fx.T
            norm1 = np.linalg.norm(r_dash, 2, 0)
            norm2 = np.linalg.norm(np.linalg.solve(self.G, u), 2, 0)
            mse = self.err_var * np.atleast_2d(1 + norm2 - norm1).T
            if self.normalize:
                mse = self.value_std * np.sqrt(mse)
            if x_.shape[1] == 1:
                mse = mse.flatten()
            return y, mse
        else:
            return y

    def jacobian(self, x):
        """
        Predict the gradient of the model at new points.

        This method evaluates the regression and correlation model at new sample point. Then, it predicts the gradient
        using the regression coefficients and the training data.

        **Input:**

        * **x** (`list` or `numpy array`):
            Points at which to evaluate the gradient.

        **Output:**

        * **grad_x** (`list` or `numpy array`):
            Gradient of the surrogate model evaluated at the new points.

        """
        x_ = np.atleast_2d(x)
        if self.normalize:
            x_ = (x_ - self.sample_mean) / self.sample_std
            s_ = (self.samples - self.sample_mean) / self.sample_std
        else:
            s_ = self.samples

        fx, jf = self.reg_model(x_)
        rx, drdx = self.corr_model(x=x_, s=s_, params=self.corr_model_params, dx=True)
        y_grad = np.einsum('ikj,jm->ik', jf, self.beta) + np.einsum('ijk,jm->ki', drdx.T, self.gamma)
        if self.normalize:
            y_grad = y_grad * self.value_std/self.sample_std
        if x_.shape[1] == 1:
            y_grad = y_grad.flatten()
        return y_grad

    # Defining Regression model (Linear)
    def _regress(self):
        if self.reg_model == 'Constant':
            def r(s):
                s = np.atleast_2d(s)
                fx = np.ones([np.size(s, 0), 1])
                jf = np.zeros([np.size(s, 0), np.size(s, 1), 1])
                return fx, jf
        elif self.reg_model == 'Linear':
            def r(s):
                s = np.atleast_2d(s)
                fx = np.concatenate((np.ones([np.size(s, 0), 1]), s), 1)
                jf_b = np.zeros([np.size(s, 0), np.size(s, 1), np.size(s, 1)])
                np.einsum('jii->ji', jf_b)[:] = 1
                jf = np.concatenate((np.zeros([np.size(s, 0), np.size(s, 1), 1]), jf_b), 2)
                return fx, jf
        else:
            def r(s):
                s = np.atleast_2d(s)
                fx = np.zeros([np.size(s, 0), int((np.size(s, 1) + 1) * (np.size(s, 1) + 2) / 2)])
                jf = np.zeros(
                    [np.size(s, 0), np.size(s, 1), int((np.size(s, 1) + 1) * (np.size(s, 1) + 2) / 2)])
                for i in range(np.size(s, 0)):
                    temp = np.hstack((1, s[i, :]))
                    for j in range(np.size(s, 1)):
                        temp = np.hstack((temp, s[i, j] * s[i, j::]))
                    fx[i, :] = temp
                    # definie H matrix
                    h_ = 0
                    for j in range(np.size(s, 1)):
                        tmp_ = s[i, j] * np.eye(np.size(s, 1))
                        t1 = np.zeros([np.size(s, 1), np.size(s, 1)])
                        t1[j, :] = s[i, :]
                        tmp = tmp_ + t1
                        if j == 0:
                            h_ = tmp[:, j::]
                        else:
                            h_ = np.hstack((h_, tmp[:, j::]))
                    jf[i, :, :] = np.hstack((np.zeros([np.size(s, 1), 1]), np.eye(np.size(s, 1)), h_))
                return fx, jf

        return r

    # Defining Correlation model (Gaussian Process)
    def _corr(self):
        def check_samples_and_return_stack(x, s):
            x_, s_ = np.atleast_2d(x), np.atleast_2d(s)
            # Create stack matrix, where each block is x_i with all s
            stack = np.tile(np.swapaxes(np.atleast_3d(x_), 1, 2), (1, np.size(s_, 0), 1)) - np.tile(s_, (
                np.size(x_, 0),
                1, 1))
            return stack

        def derivatives(x_, s_, params):
            stack = check_samples_and_return_stack(x_, s_)
            # Taking stack and creating array of all thetaj*dij
            after_parameters = params * abs(stack)
            # Create matrix of all ones to compare
            comp_ones = np.ones((np.size(x_, 0), np.size(s_, 0), np.size(s_, 1)))
            # zeta_matrix has all values min{1,theta*dij}
            zeta_matrix_ = np.minimum(after_parameters, comp_ones)
            # Copy zeta_matrix to another matrix that will used to find where derivative should be zero
            indices = zeta_matrix_.copy()
            # If value of min{1,theta*dij} is 1, the derivative should be 0.
            # So, replace all values of 1 with 0, then perform the .astype(bool).astype(int)
            # operation like in the linear example, so you end up with an array of 1's where
            # the derivative should be caluclated and 0 where it should be zero
            indices[indices == 1] = 0
            # Create matrix of all |dij| (where non zero) to be used in calculation of dR/dtheta
            dtheta_derivs_ = indices.astype(bool).astype(int) * abs(stack)
            # Same as above, but for matrix of all thetaj where non-zero
            dx_derivs_ = indices.astype(bool).astype(int) * params * np.sign(stack)
            return zeta_matrix_, dtheta_derivs_, dx_derivs_

        if self.corr_model == 'Exponential':
            def c(x, s, params, dt=False, dx=False):
                stack = check_samples_and_return_stack(x, s)
                rx = np.exp(np.sum(-params * abs(stack), axis=2))
                if dt:
                    drdt = - abs(stack) * np.transpose(np.tile(rx, (np.size(x, 1), 1, 1)), (1, 2, 0))
                    return rx, drdt
                if dx:
                    drdx = - params * np.sign(stack) * np.transpose(np.tile(rx, (np.size(x, 1), 1, 1)), (1, 2, 0))
                    return rx, drdx
                return rx
        elif self.corr_model == 'Gaussian':
            def c(x, s, params, dt=False, dx=False):
                stack = check_samples_and_return_stack(x, s)
                rx = np.exp(np.sum(-params * (stack ** 2), axis=2))
                if dt:
                    drdt = -(stack ** 2) * np.transpose(np.tile(rx, (np.size(x, 1), 1, 1)), (1, 2, 0))
                    return rx, drdt
                if dx:
                    drdx = - 2 * params * stack * np.transpose(np.tile(rx, (np.size(x, 1), 1, 1)), (1, 2, 0))
                    return rx, drdx
                return rx
        elif self.corr_model == 'Linear':
            def c(x, s, params, dt=False, dx=False):
                stack = check_samples_and_return_stack(x, s)
                # Taking stack and turning each d value into 1-theta*dij
                after_parameters = 1 - params * abs(stack)
                # Define matrix of zeros to compare against (not necessary to be defined separately,
                # but the line is bulky if this isn't defined first, and it is used more than once)
                comp_zero = np.zeros((np.size(x, 0), np.size(s, 0), np.size(s, 1)))
                # Compute matrix of max{0,1-theta*d}
                max_matrix = np.maximum(after_parameters, comp_zero)
                rx = np.prod(max_matrix, 2)
                # Create matrix that has 1s where max_matrix is nonzero
                # -Essentially, this acts as a way to store the indices of where the values are nonzero
                ones_and_zeros = max_matrix.astype(bool).astype(int)
                # Set initial derivatives as if all were positive
                first_dtheta = -abs(stack)
                first_dx = np.negative(params) * np.sign(stack)
                # Multiply derivs by ones_and_zeros...this will set the values where the
                # derivative should be zero to zero, and keep all other values the same
                drdt = np.multiply(first_dtheta, ones_and_zeros)
                drdx = np.multiply(first_dx, ones_and_zeros)
                if dt:
                    # Loop over parameters, shifting max_matrix and multiplying over derivative matrix with each iter
                    for i in range(len(params) - 1):
                        drdt = drdt * np.roll(max_matrix, i + 1, axis=2)
                    return rx, drdt
                if dx:
                    # Loop over parameters, shifting max_matrix and multiplying over derivative matrix with each iter
                    for i in range(len(params) - 1):
                        drdx = drdx * np.roll(max_matrix, i + 1, axis=2)
                    return rx, drdx
                return rx
        elif self.corr_model == 'Spherical':
            def c(x, s, params, dt=False, dx=False):
                zeta_matrix, dtheta_derivs, dx_derivs = derivatives(x_=x, s_=s, params=params)
                # Initial matrices containing derivates for all values in array. Note since
                # dtheta_s and dx_s already accounted for where derivative should be zero, all
                # that must be done is multiplying the |dij| or thetaj matrix on top of a
                # matrix of derivates w.r.t zeta (in this case, dzeta = -1.5+1.5zeta**2)
                drdt = (-1.5 + 1.5 * zeta_matrix ** 2) * dtheta_derivs
                drdx = (-1.5 + 1.5 * zeta_matrix ** 2) * dx_derivs
                # Also, create matrix for values of equation, 1 - 1.5zeta + 0.5zeta**3, for loop
                zeta_function = 1 - 1.5 * zeta_matrix + 0.5 * zeta_matrix ** 3
                rx = np.prod(zeta_function, 2)
                if dt:
                    # Same as previous example, loop over zeta matrix by shifting index
                    for i in range(len(params) - 1):
                        drdt = drdt * np.roll(zeta_function, i + 1, axis=2)
                    return rx, drdt
                if dx:
                    # Same as previous example, loop over zeta matrix by shifting index
                    for i in range(len(params) - 1):
                        drdx = drdx * np.roll(zeta_function, i + 1, axis=2)
                    return rx, drdx
                return rx
        elif self.corr_model == 'Cubic':
            def c(x, s, params, dt=False, dx=False):
                zeta_matrix, dtheta_derivs, dx_derivs = derivatives(x_=x, s_=s, params=params)
                # Initial matrices containing derivates for all values in array. Note since
                # dtheta_s and dx_s already accounted for where derivative should be zero, all
                # that must be done is multiplying the |dij| or thetaj matrix on top of a
                # matrix of derivates w.r.t zeta (in this case, dzeta = -6zeta+6zeta**2)
                drdt = (-6 * zeta_matrix + 6 * zeta_matrix ** 2) * dtheta_derivs
                drdx = (-6 * zeta_matrix + 6 * zeta_matrix ** 2) * dx_derivs
                # Also, create matrix for values of equation, 1 - 3zeta**2 + 2zeta**3, for loop
                zeta_function_cubic = 1 - 3 * zeta_matrix ** 2 + 2 * zeta_matrix ** 3
                rx = np.prod(zeta_function_cubic, 2)
                if dt:
                    # Same as previous example, loop over zeta matrix by shifting index
                    for i in range(len(params) - 1):
                        drdt = drdt * np.roll(zeta_function_cubic, i + 1, axis=2)
                    return rx, drdt
                if dx:
                    # Same as previous example, loop over zeta matrix by shifting index
                    for i in range(len(params) - 1):
                        drdx = drdx * np.roll(zeta_function_cubic, i + 1, axis=2)
                    return rx, drdx
                return rx
        else:
            def c(x, s, params, dt=False, dx=False):
                # x_, s_ = np.atleast_2d(x_), np.atleast_2d(s_)
                # # Create stack matrix, where each block is x_i with all s
                # stack = np.tile(np.swapaxes(np.atleast_3d(x_), 1, 2), (1, np.size(s_, 0), 1)) - np.tile(s_, (
                #     np.size(x_, 0),
                #     1, 1))
                stack = check_samples_and_return_stack(x, s)
                # In this case, the zeta value is just abs(stack)*parameters, no comparison
                zeta_matrix = abs(stack) * params
                # So, dtheta and dx are just |dj| and theta*sgn(dj), respectively
                dtheta_derivs = abs(stack)
                # dx_derivs = np.ones((np.size(x,0),np.size(s,0),np.size(s,1)))*parameters
                dx_derivs = np.sign(stack) * params

                # Initialize empty sigma and dsigma matrices
                sigma = np.ones((zeta_matrix.shape[0], zeta_matrix.shape[1], zeta_matrix.shape[2]))
                dsigma = np.ones((zeta_matrix.shape[0], zeta_matrix.shape[1], zeta_matrix.shape[2]))

                # Loop over cases to create zeta_matrix and subsequent dR matrices
                for i in range(zeta_matrix.shape[0]):
                    for j in range(zeta_matrix.shape[1]):
                        for k in range(zeta_matrix.shape[2]):
                            y = zeta_matrix[i, j, k]
                            if 0 <= y <= 0.2:
                                sigma[i, j, k] = 1 - 15 * y ** 2 + 30 * y ** 3
                                dsigma[i, j, k] = -30 * y + 90 * y ** 2
                            elif 0.2 < y < 1.0:
                                sigma[i, j, k] = 1.25 * (1 - y) ** 3
                                dsigma[i, j, k] = 3.75 * (1 - y) ** 2 * -1
                            elif y >= 1:
                                sigma[i, j, k] = 0
                                dsigma[i, j, k] = 0

                rx = np.prod(sigma, 2)

                if dt:
                    # Initialize derivative matrices incorporating chain rule
                    drdt = dsigma * dtheta_derivs
                    # Loop over to create proper matrices
                    for i in range(len(params) - 1):
                        drdt = drdt * np.roll(sigma, i + 1, axis=2)
                    return rx, drdt
                if dx:
                    # Initialize derivative matrices incorporating chain rule
                    drdx = dsigma * dx_derivs
                    # Loop over to create proper matrices
                    for i in range(len(params) - 1):
                        drdx = drdx * np.roll(sigma, i + 1, axis=2)
                    return rx, drdx
                return rx
        return c

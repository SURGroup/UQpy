import logging
import numpy as np
from scipy.linalg import cholesky, cho_solve

from beartype import beartype

from UQpy.utilities.Utilities import process_random_state
from UQpy.surrogates.baseclass.Surrogate import Surrogate
from UQpy.utilities.ValidationTypes import RandomStateType
from UQpy.utilities.kernels.baseclass.Kernel import Kernel
from UQpy.surrogates.gaussian_process.constraints.baseclass.Constraints import ConstraintsGPR


class GaussianProcessRegression(Surrogate):
    @beartype
    def __init__(
            self,
            kernel: Kernel,
            hyperparameters: list,
            regression_model=None,
            optimizer=None,
            bounds=None,
            optimize_constraints: ConstraintsGPR = None,
            optimizations_number: int = 1,
            normalize: bool = False,
            noise: bool = False,
            random_state: RandomStateType = None,
    ):
        """
        GaussianProcessRegressor an Gaussian process regression-based surrogate model to predict the model output at
        new sample points.

        :param kernel: `kernel` specifies and evaluates the kernel.
         Built-in options: RBF
        :param hyperparameters: List or array of initial values for the kernel hyperparameters/scale parameters. This
         input attribute defines the dimension of input training point, thus its length/shape should be equal to the
         input dimension plus one (d+1), this list/array includes 'd' length scale and process standard deviation. In
         case of noisy observations/output, its length/shape should be equal to the input dimension plus two (d+2), this
         list/array includes 'd' lengthscales, process standard deviation and noise variance.
        :param regression_model: A class object, which computes the basis function at a sample point. If
         regression_model is None, this class will train GP with regression.
         Default: None
        :param bounds: This input attribute defines the bounds of the loguniform distributions, which randomly generates
         new starting point for optimization problem to estimate maximum likelihood estimator. This should be a closed
         bound.
         Default: [10**-3, 10**3] for each hyperparameter and [10**-10, 10**-1] for variance in noise (if applicable).
        :param optimizations_number: Number of times MLE optimization problem is to be solved with a random starting
         point. Default: 1.
        :param normalize: Boolean flag used in case data normalization is required.
        :param optimizer: A class object of 'MinimizeOptimizer' or 'FminCobyla' from UQpy.utilities module. If optimizer
         is not defined, this class will not solve MLE problem and predictions will be based on hyperparameters provided
         as input.
         Default: None.
        :param noise: Boolean flag used in case of noisy training data.
         Default: False
        :param random_state: Random seed used to initialize the pseudo-random number generator. If an integer is
         provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise, the
         object itself can be passed directly.
        """
        self.regression_model = regression_model
        self.kernel = kernel
        self.hyperparameters = np.array(hyperparameters)
        self.bounds = bounds
        self.optimizer = optimizer
        self.optimizations_number = optimizations_number
        self.optimize_constraints = optimize_constraints
        self.normalize = normalize
        self.noise = noise
        self.logger = logging.getLogger(__name__)
        self.random_state = random_state

        # Variables are used outside the __init__
        self.samples = None
        self.values = None
        self.sample_mean, self.sample_std = None, None
        self.value_mean, self.value_std = None, None
        self.rmodel, self.cmodel = None, None
        self.beta = None
        """Regression coefficients."""
        self.gamma = None
        self.err_var = None
        """Variance of the Gaussian random process."""
        self.F_dash = None
        self.C_inv = None
        """Inverse Cholesky decomposition of the correlation matrix."""
        self.G = None
        self.F, self.K = None, None
        self.cc, self.alpha_ = None, None
        self.mu = 0

        if bounds is None:
            if isinstance(self.optimizer, type(None)) or isinstance(self.optimizer._bounds, type(None)):
                self._define_bounds()
            else:
                self.bounds = self.optimizer._bounds

        self.jac = False
        self.random_state = process_random_state(random_state)

    def _define_bounds(self):
        bounds_ = [[10 ** -3, 10 ** 3]] * self.hyperparameters.shape[0]
        if self.noise:
            bounds_[-1] = [10 ** -10, 10 ** -1]
        self.bounds = bounds_

    def fit(
            self,
            samples,
            values,
            optimizations_number=None,
            hyperparameters=None,
    ):
        """
        Fit the surrogate model using the training samples and the corresponding model values.

        The user can run this method multiple time after initiating the :class:`.GaussianProcessRegressor` class object.

        This method updates the samples and parameters of the :class:`.GaussianProcessRegressor` object. This method
        uses `hyperparameters` from previous run as the starting point for MLE problem unless user provides a new
        starting point.

        :param samples: `ndarray` containing the training points.
        :param values: `ndarray` containing the model evaluations at the training points.
        :param optimizations_number: number of optimization iterations
        :param hyperparameters: List or array of initial values for the kernel hyperparameters/scale parameters.

        The :meth:`fit` method has no returns, although it creates the `beta`, `err_var` and `C_inv` attributes of the
        :class:`.GaussianProcessRegressor` class.
        """
        self.logger.info("UQpy: Running gpr.fit")

        if optimizations_number is not None:
            self.optimizations_number = optimizations_number
        if hyperparameters is not None:
            self.hyperparameters = np.array(hyperparameters)
        self.samples = np.array(samples)

        # Number of samples and dimensions of samples and values
        nsamples, input_dim = self.samples.shape
        output_dim = int(np.size(values) / nsamples)

        # Verify the length of hyperparameters and input dimension
        if self.noise:
            if self.hyperparameters.shape[0] != input_dim + 2:
                raise RuntimeError("UQpy: The length/shape of attribute 'hyperparameter' and input dimension are not "
                                   "consistent.")
        elif self.hyperparameters.shape[0] != input_dim + 1:
            raise RuntimeError("UQpy: The length/shape of attribute 'hyperparameter' and input dimension are not "
                               "consistent.")

        self.values = np.array(values).reshape(nsamples, output_dim)

        # Normalizing the data
        if self.normalize:
            self.sample_mean, self.sample_std = np.mean(self.samples, 0), np.std(self.samples, 0)
            self.value_mean, self.value_std = np.mean(self.values, 0), np.std(self.values, 0)
            s_ = (self.samples - self.sample_mean) / self.sample_std
            y_ = (self.values - self.value_mean) / self.value_std
        else:
            s_ = self.samples
            y_ = self.values

        if self.regression_model is not None:
            self.F = self.regression_model.r(s_)

        # Maximum Likelihood Estimation : Solving optimization problem to calculate hyperparameters
        if self.optimizer is not None:
            # To ensure that cholesky of covariance matrix is computed again during optimization
            self.alpha_ = None
            lb = [np.log10(xy[0]) for xy in self.bounds]
            ub = [np.log10(xy[1]) for xy in self.bounds]

            starting_point = np.random.uniform(low=lb, high=ub, size=(self.optimizations_number, len(self.bounds)))
            starting_point[0, :] = np.log10(self.hyperparameters)

            if self.optimize_constraints is not None:
                cons = self.optimize_constraints.define_arguments(self.samples, self.values, self.predict)
                self.optimizer.apply_constraints(constraints=cons)
                self.optimizer.apply_constraints_argument(self.optimize_constraints.constraint_args)
            else:
                log_bounds = [[np.log10(xy[0]), np.log10(xy[1])] for xy in self.bounds]
                self.optimizer.update_bounds(bounds=log_bounds)

            minimizer = np.zeros([self.optimizations_number, len(self.bounds)])
            fun_value = np.zeros([self.optimizations_number, 1])

            for i__ in range(self.optimizations_number):
                p_ = self.optimizer.optimize(function=GaussianProcessRegression.log_likelihood,
                                             initial_guess=starting_point[i__, :],
                                             args=(self.kernel, s_, y_, self.noise, self.F),
                                             jac=self.jac)

                if isinstance(p_, np.ndarray):
                    minimizer[i__, :] = p_
                    fun_value[i__, 0] = GaussianProcessRegression.log_likelihood(p_, self.kernel, s_, y_,
                                                                                 self.noise, self.F)
                else:
                    minimizer[i__, :] = p_.x
                    fun_value[i__, 0] = p_.fun

            if min(fun_value) == np.inf:
                raise NotImplementedError("Maximum likelihood estimator failed: Choose different starting point or "
                                          "increase nopt")
            t = np.argmin(fun_value)
            self.hyperparameters = 10 ** minimizer[t, :]

        # Updated Correlation matrix corresponding to MLE estimates of hyperparameters
        if self.noise:
            self.kernel.kernel_parameter = self.hyperparameters[:-2]
            sigma = self.hyperparameters[-2]
            self.K = sigma ** 2 * self.kernel.calculate_kernel_matrix(x=s_, s=s_) + \
                     np.eye(nsamples) * (self.hyperparameters[-1]) ** 2
        else:
            self.kernel.kernel_parameter = self.hyperparameters[:-1]
            sigma = self.hyperparameters[-1]
            self.K = sigma ** 2 * self.kernel.calculate_kernel_matrix(x=s_, s=s_)

        self.cc = cholesky(self.K + 1e-10 * np.eye(nsamples), lower=True)
        self.alpha_ = cho_solve((self.cc, True), y_)

        if self.regression_model is not None:
            # Compute the regression coefficient (solving this linear equation: F * beta = Y)
            # Eq: 3.8, DACE
            f_dash = np.linalg.solve(self.cc, self.F)
            y_dash = np.linalg.solve(self.cc, y_)
            q_, g_ = np.linalg.qr(f_dash)  # Eq: 3.11, DACE
            # Check if F is a full rank matrix
            if np.linalg.matrix_rank(g_) != min(np.size(self.F, 0), np.size(self.F, 1)):
                raise NotImplementedError("Chosen regression functions are not sufficiently linearly independent")
            # Design parameters (beta: regression coefficient)
            self.beta = np.linalg.solve(g_, np.matmul(np.transpose(q_), y_dash))
            self.mu = np.einsum("ij,jk->ik", self.F, self.beta)
            self.alpha_ = cho_solve((self.cc, True), y_ - self.mu)

        self.logger.info("UQpy: gpr fit complete.")

    def predict(self, points, return_std: bool = False, hyperparameters: list = None):
        """
        Predict the model response at new points.

        This method evaluates the regression and correlation model at new sample points. Then, it predicts the function
        value and standard deviation.

        :param points: Points at which to predict the model response.
        :param return_std: Indicator to estimate standard deviation.
        :param hyperparameters: Hyperparameters for correlation model.
        :return: Predicted values at the new points, Standard deviation of predicted values at the new points
        """
        x_ = np.atleast_2d(points)
        if self.normalize:
            x_ = (x_ - self.sample_mean) / self.sample_std
            s_ = (self.samples - self.sample_mean) / self.sample_std
            y_ = (self.values - self.value_mean) / self.value_std
        else:
            s_ = self.samples
            y_ = self.values

        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        noise_std = 0
        kernelparameters = hyperparameters
        if self.noise:
            kernelparameters = hyperparameters[:-1]
            noise_std = hyperparameters[-1]

        if hyperparameters is not None:
            # This is used for MLE constraints, if constraints call 'predict' method.
            self.kernel.kernel_parameter = kernelparameters[:-1]
            sigma = kernelparameters[-1]
            K = sigma ** 2 * self.kernel.calculate_kernel_matrix(x=s_, s=s_) + \
                np.eye(self.samples.shape[0]) * (noise_std ** 2)
            cc = np.linalg.cholesky(K + 1e-10 * np.eye(self.samples.shape[0]))
            mu = 0
            if self.regression_model is not None:
                f_dash = np.linalg.solve(cc, self.F)
                y_dash = np.linalg.solve(cc, y_)
                q_, g_ = np.linalg.qr(f_dash)  # Eq: 3.11, DACE
                # Check if F is a full rank matrix
                if np.linalg.matrix_rank(g_) != min(np.size(self.F, 0), np.size(self.F, 1)):
                    raise NotImplementedError("Chosen regression functions are not sufficiently linearly independent")
                # Design parameters (beta: regression coefficient)
                beta = np.linalg.solve(g_, np.matmul(np.transpose(q_), y_dash))
                mu = np.einsum("ij,jk->ik", self.F, beta)
            alpha_ = cho_solve((cc, True), y_ - mu)
        else:
            cc, alpha_ = self.cc, self.alpha_

        mu1 = 0
        if self.regression_model is not None:
            fx = self.regression_model.r(x_)
            if self.beta is None:
                mu1 = np.einsum("ij,jk->ik", fx, beta)
            else:
                mu1 = np.einsum("ij,jk->ik", fx, self.beta)

        self.kernel.kernel_parameter = kernelparameters[:-1]
        sigma = kernelparameters[-1]

        k = sigma**2*self.kernel.calculate_kernel_matrix(x=x_, s=s_)
        y = mu1 + k @ alpha_
        if self.normalize:
            y = self.value_mean + y * self.value_std
        if x_.shape[1] == 1:
            y = y.flatten()

        if return_std:
            self.kernel.kernel_parameter = kernelparameters[:-1]
            sigma = kernelparameters[-1]
            k1 = sigma**2*self.kernel.calculate_kernel_matrix(x=x_, s=x_)
            var = (k1 - k @ cho_solve((cc, True), k.T)).diagonal()
            mse = np.sqrt(var)
            if self.normalize:
                mse = self.value_std * mse
            if x_.shape[1] == 1:
                mse = mse.flatten()
            return y, mse
        else:
            return y

    @staticmethod
    def log_likelihood(p0, k_, s, y, ind_noise, fx_):
        """

        :param p0: An 1-D numpy array of hyperparameters, that are identified through MLE. The last two elements are
                   process variance and data noise, rest of the elements are length scale along each input dimension.
        :param k_: Kernel
        :param s: Input training data
        :param y: Output training data
        :param ind_noise: Boolean flag to indicate the noisy output
        :param fx_: Basis function evaluated at training points
        :return:
        """
        # Return the log-likelihood function and it's gradient. Gradient is calculated using Central Difference
        m = s.shape[0]

        if ind_noise:
            k_.kernel_parameter = 10 ** p0[:-2]
            sigma = 10 ** p0[-2]
            k__ = sigma ** 2 * k_.calculate_kernel_matrix(x=s, s=s) + np.eye(m) * (10 ** p0[-1]) ** 2
        else:
            k_.kernel_parameter = 10 ** p0[:-1]
            sigma = 10 ** p0[-1]
            k__ = sigma ** 2 * k_.calculate_kernel_matrix(x=s, s=s)
        cc = cholesky(k__ + 1e-10 * np.eye(m), lower=True)

        mu = 0
        if fx_ is not None:
            f_dash = np.linalg.solve(cc, fx_)
            y_dash = np.linalg.solve(cc, y)
            q_, g_ = np.linalg.qr(f_dash)  # Eq: 3.11, DACE
            # Check if F is a full rank matrix
            if np.linalg.matrix_rank(g_) != min(np.size(fx_, 0), np.size(fx_, 1)):
                raise NotImplementedError("Chosen regression functions are not sufficiently linearly independent")
            # Design parameters (beta: regression coefficient)
            beta = np.linalg.solve(g_, np.matmul(np.transpose(q_), y_dash))
            mu = np.einsum("ij,jk->ik", fx_, beta)

        term1 = (y - mu).T @ (cho_solve((cc, True), y - mu))
        term2 = 2 * np.sum(np.log(np.abs(np.diag(cc))))

        return 0.5 * (term1 + term2 + m * np.log(2 * np.pi))[0, 0]

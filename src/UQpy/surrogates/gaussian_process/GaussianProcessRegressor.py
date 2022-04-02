import logging
import numpy as np
from scipy.linalg import cholesky, cho_solve

from beartype import beartype


from UQpy.utilities.Utilities import process_random_state
from UQpy.surrogates.baseclass.Surrogate import Surrogate
from UQpy.utilities.ValidationTypes import RandomStateType
from UQpy.surrogates.gaussian_process.kernels.baseclass.Kernel import Kernel
from UQpy.surrogates.gaussian_process.constraints.baseclass.Constraints import ConstraintsGPR


class GaussianProcessRegressor(Surrogate):
    @beartype
    def __init__(
            self,
            # regression_model: Regression,
            kernel: Kernel,
            hyperparameters: list,
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
         input dimension (d). In case of noisy observations/output, its length/shape should be equal to the input
         dimension plus one (d+1).
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
        # self.regression_model = regression_model
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

        if bounds is None:
            if self.optimizer._bounds is None:
                bounds_ = [[10**-3, 10**3]] * self.hyperparameters.shape[0]
                if self.noise:
                    bounds_[-1] = [10**-10, 10**-1]
                self.bounds = bounds_
            else:
                self.bounds = self.optimizer._bounds

        self.jac = False
        self.random_state = process_random_state(random_state)

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

        # self.F, jf_ = self.regression_model.r(s_)

        # Maximum Likelihood Estimation : Solving optimization problem to calculate hyperparameters
        if self.optimizer is not None:
            # To ensure that cholesky of covariance matrix is computed again during optimization
            self.alpha_ = None
            lb = [np.log10(xy[0]) for xy in self.bounds]
            ub = [np.log10(xy[1]) for xy in self.bounds]

            starting_point = np.random.uniform(low=lb, high=ub, size=(self.optimizations_number, len(self.bounds)))
            starting_point[0, :] = np.log10(self.hyperparameters)

            if self.optimize_constraints is not None:
                cons = self.optimize_constraints.constraints(self.samples, self.values, self.predict)
                self.optimizer.apply_constraints(constraints=cons)
                self.optimizer.apply_constraints_argument(self.optimize_constraints.args)
            else:
                log_bounds = [[np.log10(xy[0]), np.log10(xy[1])] for xy in self.bounds]
                self.optimizer.update_bounds(bounds=log_bounds)

            minimizer = np.zeros([self.optimizations_number, len(self.bounds)])
            fun_value = np.zeros([self.optimizations_number, 1])

            for i__ in range(self.optimizations_number):
                p_ = self.optimizer.optimize(function=GaussianProcessRegressor.log_likelihood,
                                             initial_guess=starting_point[i__, :],
                                             args=(self.kernel, s_, y_, self.noise),
                                             jac=self.jac)

                if isinstance(p_, np.ndarray):
                    minimizer[i__, :] = p_
                    fun_value[i__, 0] = GaussianProcessRegressor.log_likelihood(p_, self.kernel, s_, y_, self.noise)
                else:
                    minimizer[i__, :] = p_.x
                    fun_value[i__, 0] = p_.fun

            if min(fun_value) == np.inf:
                raise NotImplementedError("Maximum likelihood estimator failed: Choose different starting point or "
                                          "increase nopt")
            t = np.argmin(fun_value)
            self.hyperparameters = 10**minimizer[t, :]

        # Updated Correlation matrix corresponding to MLE estimates of hyperparameters
        if self.noise:
            self.K = self.kernel.c(x=s_, s=s_, params=self.hyperparameters[:-1]) + \
                     np.eye(nsamples)*(self.hyperparameters[-1])**2
        else:
            self.K = self.kernel.c(x=s_, s=s_, params=self.hyperparameters)

        self.cc = cholesky(self.K + 1e-10 * np.eye(nsamples), lower=True)
        self.alpha_ = cho_solve((self.cc, True), y_)

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
        # fx, jf = self.regression_model.r(x_)

        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        noise_std = 0
        kernelparameters = hyperparameters
        if self.noise:
            kernelparameters = hyperparameters[:-1]
            noise_std = hyperparameters[-1]

        if self.alpha_ is None:
            # This is used during MLE computation
            K = self.kernel.c(x=s_, s=s_, params=kernelparameters) + \
                np.eye(self.samples.shape[0]) * (noise_std ** 2)
            cc = np.linalg.cholesky(K + 1e-10 * np.eye(self.samples.shape[0]))
            alpha_ = cho_solve((cc, True), y_)
        else:
            cc, alpha_ = self.cc, self.alpha_
        k = self.kernel.c(x=x_, s=s_, params=kernelparameters)
        y = k @ alpha_
        if self.normalize:
            y = self.value_mean + y * self.value_std
        if x_.shape[1] == 1:
            y = y.flatten()

        if return_std:
            k1 = self.kernel.c(x=x_, s=x_, params=kernelparameters)
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
    def log_likelihood(p0, k_, s, y, ind_noise):
        """

        :param p0: An 1-D numpy array of hyperparameters, that are identified through MLE. The last two elements are
                   process variance and data noise, rest of the elements are length scale along each input dimension.
        :param k_: Kernel
        :param s: Input training data
        :param y: Output training data
        :param ind_noise: Boolean flag to indicate the noisy output
        :return:
        """
        # Return the log-likelihood function and it's gradient. Gradient is calculated using Central Difference
        m = s.shape[0]
        if ind_noise:
            k__ = k_.c(x=s, s=s, params=10 ** p0[:-1]) + np.eye(m) * (10 ** p0[-1]) ** 2
        else:
            k__ = k_.c(x=s, s=s, params=10 ** p0)
        cc = cholesky(k__ + 1e-10 * np.eye(m), lower=True)

        term1 = y.T @ (cho_solve((cc, True), y))
        term2 = 2 * np.sum(np.log(np.abs(np.diag(cc))))

        return 0.5 * (term1 + term2 + m * np.log(2 * np.pi))[0, 0]

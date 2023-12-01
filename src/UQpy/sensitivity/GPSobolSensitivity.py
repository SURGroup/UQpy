import os
import logging
from typing import Union, Callable
from beartype import beartype
import matplotlib.pyplot as plt
from UQpy.distributions import *
from joblib import Parallel, delayed
from scipy.integrate import quad, dblquad
from line_profiler_pycharm import profile
from UQpy.utilities.ValidationTypes import *
from UQpy.sampling import MonteCarloSampling
from UQpy.surrogates.baseclass import Surrogate
from sklearn.gaussian_process import GaussianProcessRegressor
SurrogateType = Union[Surrogate, GaussianProcessRegressor,
                      Annotated[object, Is[lambda x: hasattr(x, 'fit') and hasattr(x, 'predict')]]]


class GPSobolSensitivity:
    """
    The class for computing first order Sobol Indices.

    **Inputs:**

    * **samples** (`ndarray`):
        A numpy array containing the data set used to train the surrgate model.

    * **surrogate** (`class` object):
        A object defining a GP surrogate model, this object must have ``fit`` and ``predict`` methods.

        This can be an object of the ``UQpy`` ``GaussianProcessRegression`` class or an object of the ``scikit-learn``
        ``GaussianProcessRegressor``

    * **distributions** ((list of) ``Distribution`` object(s)):
        List of ``Distribution`` objects corresponding to each random variable.

    * **mcs_object** (`class` object):
        A class object of UQpy.SampleMethods.MCS class to compute a monte carlo estimate of the output variance using
        surrogate's prediction method.

    * **single_integral** (`callable`):
        A method to compute the single integration of correlation model. If None, a numerical estimate is identified
        using `scipy.integrate.quad`.

    * **double_integral** (`callable`):
        A method to compute the double integration of correlation model. If None, a numerical estimate is identified
        using `scipy.integrate.dblquad`.

    * **step_size** (`float`)
        Defines the size of the step to use for gradient estimation using central difference method.

        Used only in gradient-enhanced refined stratified sampling.

    * **n_candidates** (`int`):
        Number of candidate points along each dimension, randomly generated using Latin Hypercube Sampling, to compute
        Sobol Indices.

    * **n_simulations** (`int`):
        Number of estimates of Sobol Indices to compute mean and standard deviation.

    * **lower_bound** (`float`):
        A float between 0 and 1, which defines the lower bound for integration of correlation model. The lower bound is
        computed by taking the inverse transform of the provide value.

        Eg: If `distributions`=Uniform(loc=1, scale=1) and `lower_bound`=0.02
        then lower bound for integration is distributions.icdf(0.02) = 1.02.

        This value is used if a callable is not provided for  `single_integral` and 'double_integral' attribute.
        Default: 0.01

    * **lower_bound** (`float`):
        A float between 0 and 1, which defines the upper bound for integration of correlation model. The upper bound is
        computed by taking the inverse transform of the provide value.

        Eg: If `distributions`=Uniform(loc=1, scale=1) and `upper_bound`=0.98
        then upper bound for integration is distributions.icdf(0.98) = 1.98.

        This value is used if a callable is not provided for  `single_integral` and 'double_integral' attribute.
        Default: 0.99

    * **transform_x** (`class` object):
        A class object to transform and inverse transform the input samples. This class object should have `transform`
        and `inverse_transform` methods.

    * **transform_y** (`class` object):
        A class object to transform and inverse transform the output samples. This class object should have `transform`
        and `inverse_transform` methods.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

        Default value: False

    **Attributes:**

    Each of the above inputs are saved as attributes, in addition to the following created attributes.

        * **sobol_mean** (`ndarray`):
            The generated stratified samples following the prescribed distribution.

        * **sobol_std** (`ndarray`)
            The generated samples on the unit hypercube.

        **Methods:**
    """
    # TODO: Update doc strings, add types of attributes
    def __init__(self, surrogate: SurrogateType,
                 distributions: Union[JointIndependent, Union[list, tuple]] = None,
                 samples: Union[NumpyFloatArray, NumpyIntArray] = None,
                 mcs_object: MonteCarloSampling = None,
                 single_integral: Callable = None,
                 double_integral: Callable = None,
                 n_candidates: PositiveInteger = 1000,
                 n_simulations: PositiveInteger = 10000,
                 lower_bound: PositiveFloat = 0.001,
                 upper_bound: PositiveFloat = 0.999,
                 random_state: RandomStateType = 0,
                 n_cores: PositiveInteger = 1,
                 transform_x: callable = None,
                 transform_y: callable = None,
                 compute_var: bool = False,
                 compute_cov: bool = False,
                 **kwargs
                 ) -> None:

        # Create logger with the same name as the class
        self.logger = logging.getLogger(__name__)

        self.samples, self.samples_t = samples, None
        self.surrogate = surrogate
        self.distributions = distributions
        self.mcs_object = mcs_object
        self.single_integral = single_integral
        self.double_integral = double_integral
        self.n_candidates = n_candidates
        self.n_simulations = n_simulations
        self.lower_bound, self.upper_bound = lower_bound, upper_bound
        self.lower, self.upper = None, None
        self.dimension = 1
        self.dist_moments = None
        self.single_int_corr_f, self.double_int_corr_f = None, None
        self.mean_vec, self.cov_mat = None, None
        self.sobol_mean, self.sobol_std = None, None
        self.sobol_mean1, self.sobol_std1 = None, None
        self.discrete_samples, self.transformed_discrete_samples = None, None
        self.transform_x, self.transform_y = transform_x, transform_y
        self.kwargs = kwargs
        self.n_cores = n_cores
        self.realizations, self.sobol_estimates = None, None
        self.total_var = None
        self.compute_cov = compute_cov
        self.compute_var = compute_var
        self.sample_std = None

        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        if self.samples is not None and not isinstance(self.samples, np.ndarray):
            raise NotImplementedError("Attribute 'samples' should be a numpy array")

        if self.mcs_object is None:
            self.mcs_object = MonteCarloSampling(distributions=self.distributions, nsamples=100000,
                                                 random_state=self.random_state)

        if isinstance(self.distributions, list):
            # for i in range(len(self.distributions)):
            #     if not isinstance(self.distributions[i], DistributionContinuous1D):
            #         raise TypeError('UQpy: A DistributionContinuous1D object must be provided.')
            self.dimension = len(self.distributions)
        else:
            self.dimension = 1
            # if not isinstance(self.distributions, DistributionContinuous1D):
            #     raise TypeError('UQpy: A DistributionContinuous1D object must be provided.')

        if self.single_integral is None:
            self.single_integral = self._numerical_approx_single

        if self.double_integral is None:
            self.double_integral = self._numerical_approx_double

    @beartype
    def run(self, samples=None, values=None):
        if samples is not None and isinstance(samples, np.ndarray):
            self.samples = samples
            if self.transform_x is not None:
                self.transform_x.fit(self.samples)

        if values is not None and isinstance(values, np.ndarray):
            if self.transform_y is not None:
                self.transform_y.fit(values)

        # ################## MCS ESTIMATE OF OUTPUT VARIANCE ##################

        # Store GPR variance
        mcs_samples_t = self._transform_x(self.mcs_object.samples)
        self.total_var = np.var(self._inverse_transform_y(self.surrogate.predict(mcs_samples_t)), ddof=1)

        # ################## DEFINE SUPPORT OF MARGINAL DISTRIBUTION ##################

        self.lower, self.upper = np.zeros([1, self.dimension]), np.zeros([1, self.dimension])
        for k_ in range(self.dimension):
            self.lower[0, k_] = self.distributions[k_].icdf(self.lower_bound)
            self.upper[0, k_] = self.distributions[k_].icdf(self.upper_bound)
        self.lower = self._transform_x(self.lower).reshape(-1, )
        self.upper = self._transform_x(self.upper).reshape(-1, )

        self.corr_model_params = self.surrogate.kernel_.k2.length_scale

        # Moments about origin for Distribution Object
        self.dist_moments = np.zeros([4, len(self.distributions)])
        for k_ in range(len(self.distributions)):
            self.dist_moments[:, k_] = self.distributions[k_].moments()

        self.samples_t = self._transform_x(self.samples)

        sam_mean, sam_std = [0] * self.dimension, [1] * self.dimension
        if self.transform_x is not None:
            # Update this properly, so that transform_x and transform_y are updated after executing 'run' method
            sam_mean, sam_std = self.transform_x.mean_, self.transform_x.scale_

        # ################## COMPUTE EXPECTION OF KERNEL ##################

        # Single integration components of the correlation matrix
        self.single_int_corr_f = np.zeros_like(self.samples)

        for k_ in range(self.dimension):
            for l_ in range(self.samples.shape[0]):
                self.single_int_corr_f[l_, k_] = self.single_integral(s_t=self.samples_t[l_, k_],
                                                                      d_=self.distributions[k_],
                                                                      corr_model=self.corr_model, sam_std=sam_std,
                                                                      k__=k_, l_=self.lower[k_], u_=self.upper[k_],
                                                                      sam_mean=sam_mean, kg_=self.surrogate)

        # Double integration components of the correlation matrix
        self.double_int_corr_f = np.zeros(self.dimension)
        if self.compute_cov or self.compute_var:
            for l_ in range(self.dimension):
                self.double_int_corr_f[l_] = self.double_integral(d_=self.distributions[l_], corr_model=self.corr_model,
                                                                  sam_std=sam_std, k__=l_, l_=self.lower[l_],
                                                                  u_=self.upper[l_], sam_mean=sam_mean,
                                                                  kg_=self.surrogate)
        self.sobol_mean, self.sobol_std, self.sobol_mean1, self.sobol_std1 = [], [], [], []
        self.cov_mat = np.ones([self.dimension, self.n_candidates])
        if self.compute_cov:
            self.cov_mat = np.ones([self.dimension, self.n_candidates, self.n_candidates])
        self.mean_vec = np.ones([self.n_candidates, self.dimension])

        self.discrete_samples = MonteCarloSampling(distributions=self.distributions,
                                                   nsamples=self.n_candidates).samples
        self.transformed_discrete_samples = self._transform_x(self.discrete_samples)
        self.realizations, self.sobol_estimates = [], np.zeros([self.n_simulations, self.dimension])
        # Define input argument for parallel function
        parallel_input = []
        results = []
        for i in range(self.dimension):
            parallel_input.append([i, self.transformed_discrete_samples, self.compute_mean_vector,
                                   self.compute_cov_matrix, self.compute_cov, self.compute_var])
            if self.n_cores == 1:
                results.append(self._parallel_process(i, self.transformed_discrete_samples, self.compute_mean_vector,
                                                      self.compute_cov_matrix, self.compute_cov, self.compute_var))

        # print(self.n_cores)
        if self.n_cores > 1:
            results = Parallel(n_jobs=self.n_cores, verbose=10)(delayed(self._parallel_process)(*args) for args in
                                                                parallel_input)

        for i in range(self.dimension):
            if self.compute_cov:
                self.cov_mat[i, :, :] = results[i][1].copy()
                self.mean_vec[:, i] = results[i][0].reshape(-1, ).copy()
            elif self.compute_var:
                self.cov_mat[i, :] = results[i][1].copy()
                self.mean_vec[:, i] = results[i][0].reshape(-1, ).copy()
            else:
                self.mean_vec[:, i] = results[i].reshape(-1, ).copy()

            if self.compute_cov:
                index_x = np.argsort(self.transformed_discrete_samples[:, i])

                transformed_realizations, e1, v1 = self._generate_rv(results[i][1], results[i][0], self.n_simulations,
                                                                     index_x)
                realizations = np.zeros_like(transformed_realizations)
                for ij_ in range(self.n_simulations):
                    realizations[:, ij_] = self._inverse_transform_y(transformed_realizations[:, ij_]).reshape(-1, )

                self.realizations.append(realizations)
                self.sobol_estimates[:, i] = np.var(realizations, axis=0, ddof=1) / self.total_var

                self.sobol_mean.append(np.mean(self.sobol_estimates[:, i]))
                self.sobol_std.append(np.std(self.sobol_estimates[:, i], ddof=1))

                self.sobol_mean1.append(e1/self.total_var)
                self.sobol_std1.append(np.sqrt(v1)/self.total_var)
                print('Mean: ', self.sobol_mean, self.sobol_mean1)
                print('Std: ', self.sobol_std, self.sobol_std1)
            else:
                tmp_mean_vec = self._inverse_transform_y(self.mean_vec[:, i]).reshape(-1)
                self.sobol_mean.append(np.var(tmp_mean_vec) / self.total_var)
                # self.sobol_mean = list(np.var(self.mean_vec, axis=0).reshape(-1, ) / self.total_var)
        self.sample_std = sam_std

    @staticmethod
    def _parallel_process(i__, transformed_discrete_samples, compute_mean_vector, compute_cov_matrix, compute_cov_,
                          compute_var_):
        transformed_discrete_samples_i = transformed_discrete_samples[:, i__].copy()

        # ##Compute the mean of Gaussian process (A(X^i)=E[Y|X^i])
        # Mean vector at learning/candidate input points
        mean = compute_mean_vector(transformed_discrete_samples_i, i__)

        # ##Compute the covariance of Gaussian process (A(X^i)=E[Y|X^i])
        if compute_var_ or compute_cov_:
            cov = compute_cov_matrix(transformed_discrete_samples_i, i__)
            return mean, cov
        return mean

    def _generate_rv(self, cov_matrix, mean_vector, nsamples, idx_=None):
        if idx_ is not None:
            cov_matrix = cov_matrix[idx_][:, idx_]
            mean_vector = mean_vector[idx_]
        e_val, e_vec = np.linalg.eigh(cov_matrix)
        eigen_values = e_val
        eigen_vectors = e_vec

        # Remove negative eigenvalues
        n_positive_ev = np.sum(eigen_values > 0)
        eigen_values = np.diag(eigen_values[-n_positive_ev:])
        eigen_vectors = eigen_vectors[:, -n_positive_ev:]

        tmp = np.var(eigen_vectors, axis=0)
        exp_si = np.var(mean_vector) + np.diag(eigen_values).dot(tmp)

        tmp1 = 2 * (np.diag(eigen_values) ** 2).dot(tmp ** 2)

        centered_eig_vec = (eigen_vectors - np.mean(eigen_vectors, axis=0))

        e1e2 = np.matmul(np.diag(eigen_values).reshape(-1, 1), np.diag(eigen_values).reshape(1, -1))
        tmp2 = e1e2 * np.cov(eigen_vectors.T) ** 2
        tmp2 = 4 * np.sum(tmp2[np.triu_indices(n_positive_ev, k=1)])

        centered_mean = (mean_vector - np.mean(mean_vector)).reshape(-1, )

        tmp3 = 4 * np.diag(eigen_values).dot(np.square(centered_mean.T.dot(centered_eig_vec))) / (n_positive_ev - 1) ** 2
        var_si = tmp1 + tmp2 + tmp3

        xi = self.random_state.normal(size=(n_positive_ev, nsamples))
        realiz = np.matmul(eigen_vectors, np.matmul(np.sqrt(eigen_values), xi))
        return mean_vector + realiz, exp_si, var_si

    @profile
    def corr_model(self, x, s, dimension_index):
        tmpx, tmps = np.zeros([x.shape[0], self.dimension]), np.zeros([s.shape[0], self.dimension])
        if isinstance(dimension_index, list):
            tmpx[:, dimension_index], tmps[:, dimension_index] = x, s
        else:
            tmpx[:, dimension_index], tmps[:, dimension_index] = x.reshape(-1, ), s.reshape(-1, )
        return self.surrogate.kernel_.k2(tmpx, tmps)

    def compute_mean_vector(self, scaled_sam, i_):
        k1_val = self.surrogate.kernel_.k1.constant_value
        tmp = np.delete(self.single_int_corr_f, i_, 1)

        rx_mat = np.prod(tmp, axis=1) * self.corr_model(scaled_sam, self.samples_t[:, i_], i_)
        mean_term2 = k1_val * np.matmul(rx_mat, self.surrogate.alpha_)
        return mean_term2

    @profile
    def compute_cov_matrix(self, scaled_sam, i_):
        tmp_db = np.prod(np.delete(self.double_int_corr_f, i_))
        k1_val = self.surrogate.kernel_.k1.constant_value

        # Compute R^{-1}
        cc_inv = np.linalg.inv(self.surrogate.L_)
        r_inv = cc_inv.T.dot(cc_inv)
        scaled_sam_ = np.zeros([scaled_sam.shape[0], self.dimension])
        scaled_sam_[:, i_] = scaled_sam.reshape(scaled_sam_[:, i_].shape)

        tmp1 = np.delete(self.single_int_corr_f, i_, 1)
        scikit_corr = self.corr_model(scaled_sam, self.samples_t[:, i_], i_)
        tmp_p_mat = np.prod(tmp1, axis=1) * scikit_corr
        term2 = np.einsum('ij,ji->i', tmp_p_mat, np.matmul(r_inv, tmp_p_mat.T))

        if self.compute_cov:
            u_mat = self.corr_model(scaled_sam, scaled_sam, i_)
            cov_mat = k1_val * (np.prod(tmp_db) * u_mat - k1_val * term2)
        else:
            cov_mat = k1_val * (np.prod(tmp_db) * np.ones(term2.shape[0]) - k1_val * term2)
        return cov_mat

    def _transform_x(self, data, ind=None):
        """
        This function does the linear transformation of data, such that it is consistent with the domain of training set
        used to generate the kriging surrogate model.

        :param data: Data in the actual domain.
        :param ind: Index of random input needed to be transform.
        :return: The Linear transformation of data.
        """
        if self.transform_x is None:
            return data
        else:
            if ind is None:
                return self.transform_x.transform(data)
            else:
                if np.size(data.shape) == 2 and data.shape[1] == self.dimension:
                    return self.transform_x.transform(data)[:, ind]
                else:
                    tmp = np.zeros([data.shape[0], self.dimension])
                    if data.shape[1] == 1:
                        tmp[:, ind] = data.reshape(-1, )
                    else:
                        tmp[:, ind] = data
                    return self.transform_x.transform(tmp)[:, ind]

    def _transform_y(self, data):
        """
        This function does the linear transformation of data, such that it is consistent with the domain of training set
        used to generate the kriging surrogate model.

        :param data: Data in the actual domain.
        :return: The Linear transformation of data.
        """
        if self.transform_y is None:
            return data
        else:
            tmp_data = data.reshape(-1, 1)
            return self.transform_y.transform(tmp_data)

    def _inverse_transform_x(self, data, ind=None):
        """
        This function does the inverse linear transformation on the data, and returns the raw inputs/output.

        :param data: Transformed data.
        :param ind: Index of random input needed to be inverse transformed.
        :return: The Raw data.
        """
        if self.transform_x is None:
            return data
        else:
            if np.size(data.shape) == 2 and data.shape[1] == self.dimension:
                return self.transform_x.inverse_transform(data)[:, ind]
            else:
                tmp = np.zeros([data.shape[0], self.dimension])
                if data.shape[1] == 1:
                    tmp[:, ind] = data.reshape(-1, )
                else:
                    tmp[:, ind] = data
                return self.transform_x.inverse_transform(tmp)[:, ind]

    def _inverse_transform_y(self, data):
        """
        This function does the inverse linear transformation on the data, and returns the raw inputs/output.

        :param data: Transformed data.
        :return: The Raw data.
        """
        if self.transform_y is None:
            return data
        else:
            tmp_data = data.reshape(-1, 1)
            return self.transform_y.inverse_transform(tmp_data)

    def _numerical_approx_single(self, s_t, d_, corr_model, sam_std, k__, l_, u_, kg_, sam_mean):
        return quad(self.integrand1, l_, u_, args=(s_t, d_, corr_model, k__, sam_std,
                                                   self._inverse_transform_x))[0]

    def _numerical_approx_double(self, d_, corr_model, sam_std, k__, l_, u_, kg_, sam_mean):
        return dblquad(self.integrand2, l_, u_, lambda x: l_, lambda x: u_,
                       args=(d_, corr_model, k__, sam_std, self._inverse_transform_x))[0]

    @staticmethod
    def integrand1(x_t, s_t, d_, corr_model, i_, sam_std, inv_t):
        """
        This function returns the product of the correlation model and density value. This function is integrated over
        the domain of input dimension X.

        :param x_t: Transformed sample x.
        :param s_t: Transformed sample s.
        :param d_: Distribution object.
        :param corr_model: Correlation function
        :param sam_std: Kriging Object
        :param i_: Index of input variable.
        :param inv_t: Transform input variable such that input data has mean 0 and standard deviation 1.
        :return:
        """
        x_ = inv_t(np.atleast_2d(x_t), ind=i_)
        corr = corr_model(np.atleast_2d(x_t), np.atleast_2d(s_t), i_)[0, 0]
        return corr * d_.pdf(x_) * sam_std[i_]

    @staticmethod
    def integrand2(x_t, s_t, d_, corr_model, i_, sam_std, inv_t):
        """
        This function returns the product of the correlation model and density value. This function is integrated over
        the domain of input dimension X.

        :param x_t: Transformed sample x.
        :param s_t: Transformed sample s.
        :param d_: Distribution object.
        :param corr_model: Correlation function
        :param sam_std: Kriging object
        :param i_: Index of input variable.
        :param inv_t: Transform input variable such that input data has mean 0 and standard deviation 1.
        :return:
        """
        x_ = inv_t(np.atleast_2d(x_t), ind=i_)
        s_ = inv_t(np.atleast_2d(s_t), ind=i_)
        corr = corr_model(np.atleast_2d(x_t), np.atleast_2d(s_t), i_)[0, 0]
        return corr * d_.pdf(x_) * d_.pdf(s_) * sam_std[i_] ** 2

    def run_interaction(self):
        interaction_mean, interaction_std = np.zeros([self.dimension, self.dimension]), \
                                            np.zeros([self.dimension, self.dimension])
        int_mean_vector = {}
        for i in range(self.dimension):
            for j in range(i+1, self.dimension):
                transformed_discrete_samples_ij = self.transformed_discrete_samples[:, (i, j)].copy()

                mean = self.compute_mean_vector(transformed_discrete_samples_ij, [i, j])

                cov = self.compute_cov_matrix(transformed_discrete_samples_ij, [i, j])

                transformed_realizations_ij = self._generate_rv(cov, mean, self.n_simulations)
                realizations = np.zeros_like(transformed_realizations_ij)
                for ij_ in range(self.n_simulations):
                    realizations[:, ij_] = self._inverse_transform_y(transformed_realizations_ij[:, ij_]).reshape(-1, )

                sobol_ij_estimates = np.var(realizations, axis=0)/self.total_var

                interaction_mean[i, j] = np.mean(sobol_ij_estimates) - self.sobol_mean[i] - self.sobol_mean[j]
                interaction_std[i, j] = np.std(sobol_ij_estimates - self.sobol_estimates[:, i] -
                                               self.sobol_estimates[:, j])
                int_mean_vector['{}{}'.format(i, j)] = mean

        return int_mean_vector, interaction_mean, interaction_std

    def plot_conditional_gp(self, directory=None, actual_function=None, err_bar=None, title=None):
        """

        :param directory:
        :param actual_function: List of callables, which returns conditional GP value.
        :param title:
        :return:
        """
        for i in range(self.dimension):
            sort_index = np.argsort(self.discrete_samples[:, i])
            x_points = self.discrete_samples[:, i][sort_index]
            y_estimate = self._inverse_transform_y(self.mean_vec[:, i])[sort_index]
            plt.figure()
            if err_bar is None:
                plt.plot(x_points, y_estimate, label='Estimate')
            else:
                print(self.sample_std)
                y_err = np.sqrt(self.sample_std[i]**2 * np.diag(self.cov_mat[i, :, :])[sort_index])
                plt.errorbar(x_points, y_estimate, yerr=y_err, label='Estimate')
            if actual_function is not None and isinstance(actual_function, list):
                y_actual = actual_function[i](x_points)
                plt.plot(x_points, y_actual, label='Actual')
            if directory is not None:
                plt.savefig(os.path.join(directory, 'conditional_gp_{}.jpeg'.format(i+1)), dpi=300)
            if title is not None:
                plt.title(title+' (Dim={})'.format(i+1))
            else:
                plt.title(r'Conditional GP $E[Y|X_{}]$ (Dim={})'.format(i + 1, i + 1))
            plt.xlabel(r'Input $X_{}$'.format(i + 1))
            plt.ylabel(r'$E[Y|X_{}]$'.format(i + 1))
            plt.ylim(self._inverse_transform_y(np.min(self.mean_vec)), self._inverse_transform_y(np.max(self.mean_vec)))
            plt.legend()
            plt.show()
            plt.close()

    def plot_gp_realization(self, n_realization=None, directory=None, title=None):

        if n_realization is None:
            n_realization = self.n_simulations
        for i in range(self.dimension):
            sort_index = np.argsort(self.discrete_samples[:, i])
            x_points = self.discrete_samples[:, i][sort_index]
            fig = plt.figure()
            for i__ in range(n_realization):
                plt.scatter(x_points, self.realizations[i][:, i__][sort_index])
            plt.title('Realizations of Conditional GP (N={})'.format(self.samples.shape[0]))
            plt.xlabel('Input ($x^{}$)'.format(i + 1))
            plt.ylabel('Realization of Conditional GP $A(X^{})$'.format(i + 1))
            if directory is not None:
                plt.savefig(os.path.join(directory, 'GP_realizations_{}.jpeg'.format(i+1)), dpi=300)
            if title is not None:
                plt.title(title+' (Dim={})'.format(i+1))
            else:
                plt.title(r'Realizations of Conditional GP $E[Y|X_{}]$ (Dim={})'.format(i + 1, i + 1))
            plt.show()
            plt.clf()
            plt.close(fig)

    def hist_si_estimates(self, directory=None, title=None):
        for i in range(self.dimension):
            plt.hist(self.sobol_estimates, density=True)
            plt.xlabel('Sobol estimate: Main effect of input variable {}'.format(i + 1))
            if directory is not None:
                plt.savefig(os.path.join(directory, 'sobol_estimates_{}.jpeg'.format(i+1)), dpi=300)
            if title is not None:
                plt.title(title+' (Dim={})'.format(i+1))
            else:
                plt.title(r'Conditional GP $E[Y|X_{}]$ (Dim={})'.format(i + 1, i + 1))
            plt.show()
            plt.close()

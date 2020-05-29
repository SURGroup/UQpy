"""This module contains functionality for all the stochastic process generation supported by UQpy."""

from UQpy.Utilities import *
from UQpy.Distributions import *
from scipy.linalg import sqrtm
from scipy.stats import norm
import itertools


class SRM:
    """
    A class to simulate Stochastic Processes from a given power spectrum density based on the Spectral Representation
    Method. This class can simulate both uni-variate and multi-variate multi-dimensional Stochastic Processes. Uses
    Singular Value Decomposition as opposed to Cholesky Decomposition to be more robust with near-Positive Definite
    multi-dimensional Power Spectra.

    **Input:**
    
    :param nsamples: Number of Stochastic Processes to be generated
    :type nsamples: int

    :param power_spectrum: Power spectrum to be used for generating the samples
    :type power_spectrum: numpy.ndarray

    :param frequency_length: List of frequency discretizations across dimensions
    :type frequency_length: list

    :param number_time_intervals: List of number of time discretizations across dimensions
    :type number_time_intervals: list

    :param number_frequency_intervals: List of number of frequency discretizations across dimensions
    :type number_frequency_intervals: list

    :param case: Uni-variate or Multivariate options.
                    1. 'uni' - Uni-variate
                    2. 'multi' - Multi-variate
    :type case: str

    **Attributes:**

    :param self.phi: Random Phase angles used in the simulation
    :type: self.phi: ndarray

    :param self.number_of_dimensions: Dimension of the Stochastic process
    :type: self.number_of_dimensions: int

    :param self.number_of_variables: Number of variables in the Stochastic process
    :type: self.number_of_variables: int

    :param: samples: Generated Stochastic Process
    :rtype: samples: numpy.ndarray
    
    **Author:**

    Lohit Vandanapu
    """

    # Created by Lohit Vandanapu
    # Last Modified:04/08/2020 Lohit Vandanapu

    def __init__(self, nsamples, power_spectrum, frequency_length, number_time_intervals, number_frequency_intervals,
                 random_state=None, case='uni'):
        self.power_spectrum = power_spectrum
        self.frequency_length = frequency_length
        self.number_time_intervals = number_time_intervals
        self.number_frequency_intervals = number_frequency_intervals
        self.nsamples = nsamples
        if random_state:
            np.random.seed(random_state)
        self.case = case
        if self.case == 'uni':
            self.number_of_dimensions = len(self.power_spectrum.shape)
            self.phi = np.random.uniform(
                size=np.append(self.nsamples,
                               np.ones(self.number_of_dimensions,
                                       dtype=np.int32) * self.number_frequency_intervals)) * 2 * np.pi
            self.samples = self._simulate_uni(self.phi)
        elif self.case == 'multi':
            self.number_of_variables = self.power_spectrum.shape[0]
            self.number_of_dimensions = len(self.power_spectrum.shape[2:])
            self.phi = np.random.uniform(
                size=np.append(self.nsamples,
                               np.append(
                                   np.ones(self.number_of_dimensions, dtype=np.int32) * self.number_frequency_intervals,
                                   self.number_of_variables))) * 2 * np.pi
            self.samples = self._simulate_multi(self.phi)

    def _simulate_uni(self, phi):
        fourier_coefficient = np.exp(phi * 1.0j) * np.sqrt(
            2 ** (self.number_of_dimensions + 1) * self.power_spectrum * np.prod(self.frequency_length))
        samples = np.fft.fftn(fourier_coefficient,
                              np.ones(self.number_of_dimensions, dtype=np.int32) * self.number_time_intervals)
        samples = np.real(samples)
        samples = samples[:, np.newaxis]
        return samples

    def _simulate_multi(self, phi):
        power_spectrum = np.einsum('ij...->...ij', self.power_spectrum)
        coefficient = np.sqrt(2 ** (self.number_of_dimensions + 1)) * np.sqrt(np.prod(self.frequency_length))
        u, s, v = np.linalg.svd(power_spectrum)
        power_spectrum_decomposed = np.einsum('...ij,...j->...ij', u, np.sqrt(s))
        fourier_coefficient = coefficient * np.einsum('...ij,number_of_dimensions...j -> number_of_dimensions...i',
                                                      power_spectrum_decomposed, np.exp(phi * 1.0j))
        fourier_coefficient[np.isnan(fourier_coefficient)] = 0
        samples = np.real(
            np.fft.fftn(fourier_coefficient, s=[self.number_time_intervals for _ in range(self.number_of_dimensions)],
                        axes=tuple(np.arange(1, 1 + self.number_of_dimensions))))
        samples = np.einsum('number_of_dimensions...number_of_variables->nm...', samples)
        return samples


class BSRM:
    """
    A class to simulate Stochastic Processes from a given power spectrum and bispectrum density based on the BiSpectral
    Representation Method.This class can simulate both uni-variate and multi-variate multi-dimensional Stochastic
    Processes. This class uses Singular Value Decomposition as opposed to Cholesky Decomposition to be more robust with
    near-Positive Definite multi-dimensional Power Spectra.

    **Input:**

    :param: nsamples: Number of Stochastic Processes to be generated
    :type: nsamples: int

    :param power_spectrum: Power Spectral Density to be used for generating the samples
    :type power_spectrum: numpy.ndarray

    :param bispectrum: BiSpectral Density to be used for generating the samples
    :type bispectrum: numpy.ndarray

    :param time_duration: Array of time discretizations across dimensions
    :type time_duration: numpy.ndarray

    :param frequency_length: Array of frequency discretizations across dimensions
    :type frequency_length: numpy.ndarray

    :param number_time_intervals: Array of number of time discretizations across dimensions
    :type number_time_intervals: numpy.ndarray

    :param number_frequency_intervals: Array of number of frequency discretizations across dimensions
    :type number_frequency_intervals: numpy.ndarray

    **Attributes:**

    :param self.phi: Random Phase angles used in the simulation
    :type: self.phi: ndarray

    :param self.number_of_dimensions: Dimension of the Stochastic process
    :type: self.number_of_dimensions: int

    :param self.b_ampl: Amplitude of the complex-valued Bispectrum
    :type: self.b_ampl: ndarray

    :param self.b_real: Real part of the complex-valued Bispectrum
    :type: self.b_real: ndarray

    :param self.b_imag: Imaginary part of the complex-valued Bispectrum
    :type: self.b_imag: ndarray

    :param self.biphase: biphase values of the complex-valued Bispectrum
    :type: self.biphase: ndarray

    :param self.Bc2: Values of the squared Bicoherence
    :type: self.Bc2: ndarray

    :param self.sum_Bc2: Values of the sum of squared Bicoherence
    :type: self.sum_Bc2: ndarray

    :param self.PP: Pure part of the Power Spectrum
    :type: self.PP: ndarray

    **Output:**

    :param: samples: Generated Stochastic Process
    :rtype samples: numpy.ndarray

    **Author:**

    Lohit Vandanapu
    """

    # Created by Lohit Vandanapu
    # Last Modified:04/08/2020 Lohit Vandanapu

    def __init__(self, nsamples, power_spectrum, bispectrum, time_duration, frequency_length, number_time_intervals,
                 number_frequency_intervals, random_state=None, case='uni'):
        self.nsamples = nsamples
        self.number_frequency_intervals = number_frequency_intervals
        self.number_time_intervals = number_time_intervals
        self.frequency_length = frequency_length
        self.time_duration = time_duration
        self.number_of_dimensions = len(power_spectrum.shape)
        self.power_spectrum = power_spectrum
        self.bispectrum = bispectrum
        if random_state: np.random.seed(random_state)
        self.b_ampl = np.absolute(bispectrum)
        self.b_real = np.real(bispectrum)
        self.b_imag = np.imag(bispectrum)
        self.biphase = np.arctan2(self.b_imag, self.b_real)
        self.biphase[np.isnan(self.biphase)] = 0
        self.phi = np.random.uniform(size=np.append(self.nsamples, np.ones(self.number_of_dimensions,
                                                                           dtype=np.int32) * self.number_frequency_intervals)) * 2 * np.pi
        self._compute_bicoherence()
        self.samples = self._simulate_bsrm_uni()

    def _compute_bicoherence(self):
        self.Bc2 = np.zeros_like(self.b_real)
        self.PP = np.zeros_like(self.power_spectrum)
        self.sum_Bc2 = np.zeros_like(self.power_spectrum)

        if self.number_of_dimensions == 1:
            self.PP[0] = self.power_spectrum[0]
            self.PP[1] = self.power_spectrum[1]

        if self.number_of_dimensions == 2:
            self.PP[0, :] = self.power_spectrum[0, :]
            self.PP[1, :] = self.power_spectrum[1, :]
            self.PP[:, 0] = self.power_spectrum[:, 0]
            self.PP[:, 1] = self.power_spectrum[:, 1]

        if self.number_of_dimensions == 3:
            self.PP[0, :, :] = self.power_spectrum[0, :, :]
            self.PP[1, :, :] = self.power_spectrum[1, :, :]
            self.PP[:, 0, :] = self.power_spectrum[:, 0, :]
            self.PP[:, 1, :] = self.power_spectrum[:, 1, :]
            self.PP[:, :, 0] = self.power_spectrum[:, :, 0]
            self.PP[:, 0, 1] = self.power_spectrum[:, :, 1]

        self.ranges = [range(self.number_frequency_intervals) for _ in range(self.number_of_dimensions)]

        for i in itertools.product(*self.ranges):
            wk = np.array(i)
            for j in itertools.product(*[range(k) for k in np.int32(np.ceil((wk + 1) / 2))]):
                wj = np.array(j)
                wi = wk - wj
                if self.b_ampl[(*wi, *wj)] > 0 and self.PP[(*wi, *[])] * self.PP[(*wj, *[])] != 0:
                    self.Bc2[(*wi, *wj)] = self.b_ampl[(*wi, *wj)] ** 2 / (
                            self.PP[(*wi, *[])] * self.PP[(*wj, *[])] * self.power_spectrum[
                        (*wk, *[])]) * self.frequency_length ** self.number_of_dimensions
                    self.sum_Bc2[(*wk, *[])] = self.sum_Bc2[(*wk, *[])] + self.Bc2[(*wi, *wj)]
                else:
                    self.Bc2[(*wi, *wj)] = 0
            if self.sum_Bc2[(*wk, *[])] > 1:
                print('Results may not be as expected as sum of partial bicoherences is greater than 1')
                for j in itertools.product(*[range(k) for k in np.int32(np.ceil((wk + 1) / 2))]):
                    wj = np.array(j)
                    wi = wk - wj
                    self.Bc2[(*wi, *wj)] = self.Bc2[(*wi, *wj)] / self.sum_Bc2[(*wk, *[])]
                self.sum_Bc2[(*wk, *[])] = 1
            self.PP[(*wk, *[])] = self.power_spectrum[(*wk, *[])] * (1 - self.sum_Bc2[(*wk, *[])])

    def _simulate_bsrm_uni(self):
        Coeff = np.sqrt((2 ** (
                self.number_of_dimensions + 1)) * self.power_spectrum * self.frequency_length ** self.number_of_dimensions)
        Phi_e = np.exp(self.phi * 1.0j)
        Biphase_e = np.exp(self.biphase * 1.0j)
        B = np.sqrt(1 - self.sum_Bc2) * Phi_e
        Bc = np.sqrt(self.Bc2)

        Phi_e = np.einsum('i...->...i', Phi_e)
        B = np.einsum('i...->...i', B)

        for i in itertools.product(*self.ranges):
            wk = np.array(i)
            for j in itertools.product(*[range(k) for k in np.int32(np.ceil((wk + 1) / 2))]):
                wj = np.array(j)
                wi = wk - wj
                B[(*wk, *[])] = B[(*wk, *[])] + Bc[(*wi, *wj)] * Biphase_e[(*wi, *wj)] * Phi_e[(*wi, *[])] * \
                                Phi_e[(*wj, *[])]

        B = np.einsum('...i->i...', B)
        Phi_e = np.einsum('...i->i...', Phi_e)
        B = B * Coeff
        B[np.isnan(B)] = 0
        samples = np.fft.fftn(B, [self.number_time_intervals for _ in range(self.number_of_dimensions)])
        samples = samples[:, np.newaxis]
        return np.real(samples)


class KLE:
    """
    A class to simulate Stochastic Processes from a given auto-correlation function based on the Karhunen-Louve
    Expansion

    **Input:**

    :param: nsamples: Number of Stochastic Processes to be generated
    :type: nsamples: int

    :param: correlation_function: Correlation Function to be used for generating the samples
    :type: correlation_function: numpy.ndarray

    **Attributes:**

    :param: number_eigen_values: Number of eigen values to be used in the expansion
    :type: number_eigen_values: int

    :param: samples: Generated Stochastic Process
    :rtype: samples: numpy.ndarray

    **Author:**

    Lohit Vandanapu
    """

    # Created by Lohit Vandanapu
    # Last Modified:04/08/2020 Lohit Vandanapu

    def __init__(self, nsamples, correlation_function, time_duration, threshold=None, random_state=None):
        self.correlation_function = correlation_function
        self.time_duration = time_duration
        if threshold:
            self.number_eigen_values = threshold
        else:
            self.number_eigen_values = len(self.correlation_function[0])
        if random_state:
            np.random.seed(random_state)
        self.samples = self._simulate(nsamples)

    def _simulate(self, nsamples):
        lam, phi = np.linalg.eig(self.correlation_function)
        xi = np.random.normal(size=(self.number_eigen_values, nsamples))
        lam = np.diag(lam)
        lam = lam.astype(np.float64)
        samples = np.dot(phi[:, :self.number_eigen_values], np.dot(sqrtm(lam[:self.number_eigen_values]), xi))
        samples = np.real(samples)
        samples = samples.T
        samples = samples[:, np.newaxis]
        return samples


class Translation:
    """
    A class to translate Gaussian Stochastic Processes to non-Gaussian Stochastic Processes

    **Input:**

    :param: samples_gaussian: Gaussian Stochastic Processes
    :rtype: samples_gaussian: numpy.ndarray

    :param: auto_correlation_function_gaussian: Auto-covariance of the Gaussian Stochastic Processes
    :rtype: auto_correlation_function_gaussian: numpy.ndarray

    :param: power_spectrum_gaussian: Power Spectrum of the Gaussian Stochastic Processes
    :rtype: power_spectrum_gaussian: numpy.ndarray

    :param: distribution: UQpy Distribution object
    :rtype: distribution: UQpy.Distribution

    **Output:**

    :param: samples_non_gaussian: Non-Gaussian Stochastic Processes
    :type: samples_non_gaussian: numpy.ndarray

    :param: S_ng: Power Spectrum of the Gaussian Non-Stochastic Processes
    :type: S_ng: numpy.ndarray

    :param: auto_correlation_function_non_gaussian: Auto-covariance Function of the Non-Gaussian Stochastic Processes
    :type: auto_correlation_function_non_gaussian: numpy.ndarray

    **Author:**

    Lohit Vandanapu
    """

    # Created by Lohit Vandanapu
    # Last Modified:04/08/2020 Lohit Vandanapu

    def __init__(self, distribution, time_duration, frequency_interval, number_time_intervals, number_frequency_intervals, power_spectrum_gaussian=None, auto_correlation_function_gaussian=None, samples_gaussian=None):
        self.distribution = distribution
        if auto_correlation_function_gaussian and power_spectrum_gaussian is None:
            print('Either the Power Spectrum or the Autocorrelation function should be specified')
        if auto_correlation_function_gaussian is None:
            self.power_spectrum_gaussian = power_spectrum_gaussian
            self.auto_correlation_function_gaussian = S_to_R(power_spectrum_gaussian, np.arange(0, number_frequency_intervals) * frequency_interval, np.arange(0, number_time_intervals) * time_duration)
        elif power_spectrum_gaussian is None:
            self.auto_correlation_function_gaussian = auto_correlation_function_gaussian
            self.power_spectrum_gaussian = R_to_S(auto_correlation_function_gaussian, np.arange(0, number_frequency_intervals) * frequency_interval, np.arange(0, number_time_intervals) * time_duration)
        self.shape = self.auto_correlation_function_gaussian.shape
        self.dim = len(self.auto_correlation_function_gaussian.shape)
        if samples_gaussian is not None:
            self.samples_gaussian = samples_gaussian
            self.samples_non_gaussian = self.translate_gaussian_samples()
        self.correlation_function_non_gaussian, self.auto_correlation_function_non_gaussian = self.autocorrealtion_distortion()
        self.S_ng = R_to_S(self.auto_correlation_function_non_gaussian, np.arange(0, number_frequency_intervals) * frequency_interval, np.arange(0, number_time_intervals) * time_duration)

    def translate_gaussian_samples(self):
        standard_deviation = np.sqrt(self.auto_correlation_function_gaussian[0])
        samples_cdf = norm.cdf(self.samples_gaussian, scale=standard_deviation)
        if hasattr(self.distribution, 'icdf'):
            non_gaussian_icdf = self.distribution.icdf
            samples_non_gaussian = non_gaussian_icdf(self.samples_gaussian)
        else:
            print('Distribution does not have an inverse cdf defined')
        return samples_non_gaussian

    def autocorrealtion_distortion(self):
        r_g = R_to_r(self.auto_correlation_function_gaussian)
        r_g = np.clip(r_g, -0.999, 0.999)
        r_ng = np.zeros_like(r_g)
        # for i in itertools.product(*[range(self.num) for _ in range(self.dim)]):
        #     auto_correlation_function_non_gaussian[(*i, *[])] = self.solve_integral(r_g[(*i, *[])])
        for i in itertools.product(*[range(s) for s in self.shape]):
            r_ng[i] = self.solve_integral(r_g[i])
        R_ng = r_ng * Distribution(self.marginal).moments(self.params)[1]
        return r_ng, R_ng

    def solve_integral(self, rho):
        if rho == 1.0:
            rho = 0.999
        n = 1024
        zmax = 8
        zmin = -zmax
        points, weights = np.polynomial.legendre.leggauss(n)
        points = - (0.5 * (points + 1) * (zmax - zmin) + zmin)
        weights = weights * (0.5 * (zmax - zmin))

        xi = np.tile(points, [n, 1])
        xi = xi.flatten(order='F')
        eta = np.tile(points, n)

        first = np.tile(weights, n)
        first = np.reshape(first, [n, n])
        second = np.transpose(first)

        weights2d = first * second
        w2d = weights2d.flatten()
        # tmp_f_xi = inv_cdf(self.marginal)[0](stats.norm.cdf(xi), self.params[0])
        # tmp_f_eta = inv_cdf(self.marginal)[0](stats.norm.cdf(eta), self.params[0])
        tmp_f_xi = Distribution(self.marginal).icdf(stats.norm.cdf(xi[:, np.newaxis]), self.params)
        tmp_f_eta = Distribution(self.marginal).icdf(stats.norm.cdf(eta[:, np.newaxis]), self.params)
        coef = tmp_f_xi[:, 0] * tmp_f_eta[:, 0] * w2d
        rho_non = np.sum(coef * bi_variate_normal_pdf(xi, eta, rho))
        rho_non = (rho_non - (Distribution(self.marginal).moments(self.params)[0]) ** 2) / \
                  Distribution(self.marginal).moments(self.params)[1]
        return rho_non


class InverseTranslation:
    """
    A class to perform Iterative Translation Approximation Method to find the underlying  Gaussian Stochastic Processes
    which upon translation would yield the necessary non-Gaussian Stochastic Processes

    **Input:**

    :param: samples_non_gaussian: Non-Gaussian Stochastic Processes
    :type: samples_non_gaussian: numpy.ndarray

    :param: S_ng: Power Spectrum of the Gaussian Non-Stochastic Processes
    :type: auto_correlation_function_non_gaussian: numpy.ndarray

    :param: auto_correlation_function_non_gaussian: Auto-covariance Function of the Non-Gaussian Stochastic Processes
    :type: auto_correlation_function_non_gaussian: numpy.ndarray

    :param: marginal: name of the marginal
    :type: marginal: str

    :param: params: list of parameters for the marginal
    :type: params: list

    **Output:**

    :param: samples_gaussian: Gaussian Stochastic Processes
    :rtype: samples_gaussian: numpy.ndarray

    :param: auto_correlation_function_gaussian: Auto-covariance of the Gaussian Stochastic Processes
    :rtype: auto_correlation_function_gaussian: numpy.ndarray

    :param: power_spectrum_gaussian: Power Spectrum of the Gaussian Stochastic Processes
    :rtype: power_spectrum_gaussian: numpy.ndarray

    **Author:**

    Lohit Vandanapu
    """

    # Created by Lohit Vandanapu
    # Last Modified:04/08/2020 Lohit Vandanapu

    def __init__(self, marginal, params, dt, dw, nt, nw, R_ng=None, S_ng=None, samples_ng=None):
        self.w = np.arange(0, nw) * dw
        self.t = np.arange(0, nt) * dt
        if R_ng and S_ng is None:
            print('Either the Power Spectrum or the Autocorrelation function should be specified')
        if R_ng is None:
            self.S_ng = S_ng
            self.R_ng = S_to_R(S_ng, self.w, self.t)
        elif S_ng is None:
            self.R_ng = R_ng
            self.S_ng = R_to_S(R_ng, self.w, self.t)
        self.num = self.R_ng.shape[0]
        self.dim = len(self.R_ng.shape)
        self.marginal = marginal
        self.params = params
        if samples_ng is not None:
            self.samples_ng = samples_ng
            self.samples_g = self.inverse_translate_ng_samples()
        self.S_g = self.itam()
        self.R_g = S_to_R(self.S_g, self.w, self.t)
        self.r_g = self.R_g / self.R_g[0]

    def inverse_translate_ng_samples(self):
        # samples_cdf = cdf(self.marginal)[0](self.samples_non_gaussian, self.params[0])
        # samples_gaussian = inv_cdf(['Normal'])[0](samples_cdf, [0, 1])
        samples_cdf = Distribution(dist_name=self.marginal).cdf(self.samples_ng, self.params)
        samples_g = Distribution(dist_name='normal').icdf(samples_cdf, [0, 1])
        return samples_g

    def itam(self):
        # Initial Guess
        target_s = self.S_ng
        # Iteration Conditions
        i_converge = 0
        error0 = 100
        max_iter = 10
        target_r = S_to_R(target_s, self.w, self.t)
        r_g_iterate = target_r
        s_g_iterate = target_s
        r_ng_iterate = np.zeros_like(target_r)
        s_ng_iterate = np.zeros_like(target_s)

        for ii in range(max_iter):
            r_g_iterate = S_to_R(s_g_iterate, self.w, self.t)
            # for i in itertools.product(*[range(self.num) for _ in range(self.dim)]):
            for i in range(len(target_r)):
                r_ng_iterate[i] = self.solve_integral(r_g_iterate[i] / r_g_iterate[0])
            s_ng_iterate = R_to_S(r_ng_iterate, self.w, self.t)

            # compute the relative difference between the computed NGACF & the target correlation_function(Normalized)
            err1 = np.sum((target_s - s_ng_iterate) ** 2)
            err2 = np.sum(target_s ** 2)
            error1 = 100 * np.sqrt(err1 / err2)

            if ii == max_iter or 100 * np.sqrt(err1 / err2) < 0.0005:
                i_converge = 1

            s_g_next_iterate = (target_s / s_ng_iterate) * s_g_iterate

            # Eliminate Numerical error of Upgrading Scheme
            s_g_next_iterate[s_g_next_iterate < 0] = 0

            if i_converge == 0 and ii != max_iter:
                s_g_iterate = s_g_next_iterate
                error0 = error1

        return s_g_iterate / Distribution(self.marginal).moments(self.params)[1]

    def solve_integral(self, rho):
        if rho == 1.0:
            rho = 0.999
        n = 1024
        zmax = 8
        zmin = -zmax
        points, weights = np.polynomial.legendre.leggauss(n)
        points = - (0.5 * (points + 1) * (zmax - zmin) + zmin)
        weights = weights * (0.5 * (zmax - zmin))

        xi = np.tile(points, [n, 1])
        xi = xi.flatten(order='F')
        eta = np.tile(points, n)

        first = np.tile(weights, n)
        first = np.reshape(first, [n, n])
        second = np.transpose(first)

        weights2d = first * second
        w2d = weights2d.flatten()
        tmp_f_xi = Distribution(self.marginal).icdf(stats.norm.cdf(xi[:, np.newaxis]), self.params)
        tmp_f_eta = Distribution(self.marginal).icdf(stats.norm.cdf(eta[:, np.newaxis]), self.params)
        # tmp_f_xi = inv_cdf(self.marginal)[0](stats.norm.cdf(xi), self.params[0])
        # tmp_f_eta = inv_cdf(self.marginal)[0](stats.norm.cdf(eta), self.params[0])
        coef = tmp_f_xi * tmp_f_eta * w2d
        rho_non = np.sum(coef * bi_variate_normal_pdf(xi, eta, rho))
        rho_non = (rho_non - (Distribution(self.marginal).moments(self.params)[0]) ** 2)
        return rho_non

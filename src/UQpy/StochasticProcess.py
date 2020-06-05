"""This module contains functionality for all the stochastic process generation supported by UQpy."""

from UQpy.Utilities import *
from UQpy.Distributions import *
from scipy.linalg import sqrtm
from scipy.stats import norm
import itertools


class SRM:
    """
    Perform Monte Carlo sampling (MCS) of random variables.

    **Input:**

    * **nsamples** (`int`):
        Number of samples to be generated from power spectrum.

        The ``run`` method is automatically called if `nsamples` is provided. If `nsamples` is not provided, then the
        ``SRM`` object is created but samples are not generated.

    * **power_spectrum** (`numpy.ndarray`):
        The prescribed power spectrum.

    * **time_duration** (`list or numpy.ndarray`):
        List of time discretizations across dimensions.

        The length of the list needs to be the same as the number of dimensions.

    * **frequency_interval** (`list or numpy.ndarray`):
        List of frequency discretizations across dimensions.

        The length of the list needs to be the same as the number opf dimensions.

    * **dist_object** ((list of) ``Distribution`` object(s)):
        Probability distribution of each random variable. Must be an object (or a list of objects) of the
        ``Distribution`` class.


    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (Boolean):
        A boolean declaring whether to write text to the terminal.


    **Attributes:**

    * **samples** (`ndarray` or `list`):
        Generated samples.

        If a list of ``DistributionContinuous1D`` objects is provided for ``dist_object``, then `samples` is an
        `ndarray` with ``samples.shape=(nsamples, len(dist_object))``.

        If a ``DistributionContinuous1D`` object is provided for ``dist_object`` then `samples` is an array with
        `samples.shape=(nsamples, 1)``.

        If a ``DistributionContinuousND`` object is provided for ``dist_object`` then `samples` is an array with
        ``samples.shape=(nsamples, ND)``.

        If a list of mixed ``DistributionContinuous1D`` and ``DistributionContinuousND`` objects is provided then
        `samples` is a list with ``len(samples)=nsamples`` and ``len(samples[i]) = len(dist_object)``.

    * **samplesU01** (`ndarray` (`list`)):
        Generated samples transformed to the unit hypercube.

        This attribute exists only if the ``transform_u01`` method is invoked by the user.


    **Methods**

    """

    def __init__(self, nsamples, power_spectrum, time_duration, frequency_length, number_time_intervals,
                 number_frequency_intervals, case='uni', random_state=None, verbose=False):
        self.power_spectrum = power_spectrum
        self.time_duration = np.array(time_duration)
        self.frequency_length = np.array(frequency_length)
        self.number_time_intervals = np.array(number_time_intervals)
        self.number_frequency_intervals = np.array(number_frequency_intervals)
        self.nsamples = nsamples

        # Error checks
        t_u = 2 * np.pi / (2 * self.number_frequency_intervals * self.frequency_length)
        if (self.time_duration > t_u).any():
            raise RuntimeError('UQpy: Aliasing might occur during execution')

        self.case = case
        self.verbose = verbose

        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        self.samples = None
        self.number_of_variables = None
        self.number_of_dimensions = None
        self.phi = None

        if self.case == 'uni':
            self.number_of_dimensions = len(self.power_spectrum.shape)
        elif self.case == 'multi':
            self.number_of_variables = self.power_spectrum.shape[0]
            self.number_of_dimensions = len(self.power_spectrum.shape[2:])

        # Run Spectral Representation Method
        if self.nsamples is not None:
            self.run(nsamples=self.nsamples, random_state=self.random_state)

    def run(self, nsamples, random_state=None):
        """
        Execute the random sampling in the ``MCS`` class.

        The ``run`` method is the function that performs random sampling in the ``MCS`` class. If `nsamples` is
        provided, the ``run`` method is automatically called when the ``MCS`` object is defined. The user may also call
        the ``run`` method directly to generate samples. The ``run`` method of the ``MCS`` class can be invoked many
        times and each time the generated samples are appended to the existing samples.

        ** Input:**

        * **nsamples** (`int`):
            Number of samples to be drawn from each distribution.

            If the ``run`` method is invoked multiple times, the newly generated samples will be appended to the
            existing samples.

        * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
            Random seed used to initialize the pseudo-random number generator. Default is None.

            If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
            object itself can be passed directly.

        **Output/Returns:**

        The ``run`` method has no returns, although it creates and/or appends the `samples` attribute of the ``MCS``
        class.

        """
        # Check if a random_state is provided.

        if nsamples is None:
            raise ValueError('UQpy: Stochastic Process: Number of samples must be defined.')
        if not isinstance(nsamples, int):
            raise ValueError('UQpy: Stochastic Process: nsamples should be an integer.')

        if self.verbose:
            print('UQpy: Stochastic Process: Running Spectral Representation Method.')

        samples = None

        if self.case == 'uni':
            if self.verbose:
                print('UQpy: Stochastic Process: Starting simulation of uni-variate Stochastic Processes.')
                print('UQpy: The number of dimensions is :', self.number_of_dimensions)
            self.phi = np.random.uniform(
                size=np.append(self.nsamples, np.ones(self.number_of_dimensions, dtype=np.int32)
                               * self.number_frequency_intervals)) * 2 * np.pi
            samples = self._simulate_uni(self.phi)

        elif self.case == 'multi':
            if self.verbose:
                print('UQpy: Stochastic Process: Starting simulation of multi-variate Stochastic Processes.')
                print('UQpy: Stochastic Process: The number of variables is :', self.number_of_variables)
                print('UQpy: Stochastic Process: The number of dimensions is :', self.number_of_dimensions)
            self.phi = np.random.uniform(size=np.append(self.nsamples, np.append(
                np.ones(self.number_of_dimensions, dtype=np.int32) * self.number_frequency_intervals,
                self.number_of_variables))) * 2 * np.pi
            samples = self._simulate_multi(self.phi)

        if self.samples is None:
            self.samples = samples
        else:
            self.samples = np.concatenate((self.samples, samples), axis=0)

        if self.verbose:
            print('UQpy: Stochastic Process: Spectral Representation Method Complete.')

    def _simulate_uni(self, phi):
        fourier_coefficient = np.exp(phi * 1.0j) * np.sqrt(
            2 ** (self.number_of_dimensions + 1) * self.power_spectrum * np.prod(self.frequency_length))
        samples = np.fft.fftn(fourier_coefficient, self.number_time_intervals)
        samples = np.real(samples)
        samples = samples[:, np.newaxis]
        return samples

    def _simulate_multi(self, phi):
        power_spectrum = np.einsum('ij...->...ij', self.power_spectrum)
        coefficient = np.sqrt(2 ** (self.number_of_dimensions + 1)) * np.sqrt(np.prod(self.frequency_length))
        u, s, v = np.linalg.svd(power_spectrum)
        power_spectrum_decomposed = np.einsum('...ij,...j->...ij', u, np.sqrt(s))
        fourier_coefficient = coefficient * np.einsum('...ij,n...j -> n...i',
                                                      power_spectrum_decomposed, np.exp(phi * 1.0j))
        fourier_coefficient[np.isnan(fourier_coefficient)] = 0
        samples = np.real(np.fft.fftn(fourier_coefficient, s=self.number_time_intervals,
                                      axes=tuple(np.arange(1, 1 + self.number_of_dimensions))))
        samples = np.einsum('n...m->nm...', samples)
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

    :param time_duration: List of time discretizations across dimensions
    :type time_duration: list or numpy.ndarray

    :param frequency_length: List of frequency discretizations across dimensions
    :type frequency_length: list or numpy.ndarray

    :param number_time_intervals: List of number of time discretizations across dimensions
    :type number_time_intervals: list or numpy.ndarray

    :param number_frequency_intervals: List of number of frequency discretizations across dimensions
    :type number_frequency_intervals: list or numpy.ndarray

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
                 number_frequency_intervals, random_state=None, case='uni', verbose=False):
        self.nsamples = nsamples
        self.number_frequency_intervals = np.array(number_frequency_intervals)
        self.number_time_intervals = np.array(number_time_intervals)
        self.frequency_length = np.array(frequency_length)
        self.time_duration = np.array(time_duration)
        self.number_of_dimensions = len(power_spectrum.shape)
        self.power_spectrum = power_spectrum
        self.bispectrum = bispectrum

        # Error checks
        t_u = 2 * np.pi / (2 * self.number_frequency_intervals * self.frequency_length)
        if (self.time_duration > t_u).any():
            raise RuntimeError('UQpy: Aliasing might occur during execution')

        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        self.b_ampl = np.absolute(bispectrum)
        self.b_real = np.real(bispectrum)
        self.b_imag = np.imag(bispectrum)
        self.biphase = np.arctan2(self.b_imag, self.b_real)
        self.biphase[np.isnan(self.biphase)] = 0

        self.case = case
        self.verbose = verbose

        if self.case == 'uni':
            self.number_of_dimensions = len(self.power_spectrum.shape)
            self._compute_bicoherence_uni()
        elif self.case == 'multi':
            self.number_of_variables = self.power_spectrum.shape[0]
            self.number_of_dimensions = len(self.power_spectrum.shape[2:])

        self.phi = np.random.uniform(size=np.append(self.nsamples, np.ones(self.number_of_dimensions, dtype=np.int32) *
                                                    self.number_frequency_intervals)) * 2 * np.pi
        self.samples = self._simulate_bsrm_uni()

    def _compute_bicoherence_uni(self):
        if self.verbose:
            print('UQpy: Stochastic Process: Computing the partial bicoherence values.')
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

        self.ranges = [range(self.number_frequency_intervals[i]) for i in range(self.number_of_dimensions)]

        for i in itertools.product(*self.ranges):
            wk = np.array(i)
            for j in itertools.product(*[range(k) for k in np.ceil((wk + 1) / 2, dtype=np.int32)]):
                wj = np.array(j)
                wi = wk - wj
                if self.b_ampl[(*wi, *wj)] > 0 and self.PP[(*wi, *[])] * self.PP[(*wj, *[])] != 0:
                    self.Bc2[(*wi, *wj)] = self.b_ampl[(*wi, *wj)] ** 2 / (
                            self.PP[(*wi, *[])] * self.PP[(*wj, *[])] * self.power_spectrum[(*wk, *[])]) * \
                                           self.frequency_length ** self.number_of_dimensions
                    self.sum_Bc2[(*wk, *[])] = self.sum_Bc2[(*wk, *[])] + self.Bc2[(*wi, *wj)]
                else:
                    self.Bc2[(*wi, *wj)] = 0
            if self.sum_Bc2[(*wk, *[])] > 1:
                print('UQpy: Stochastic Process: Results may not be as expected as sum of partial bicoherences is '
                      'greater than 1')
                for j in itertools.product(*[range(k) for k in np.ceil((wk + 1) / 2, dtype=np.int32)]):
                    wj = np.array(j)
                    wi = wk - wj
                    self.Bc2[(*wi, *wj)] = self.Bc2[(*wi, *wj)] / self.sum_Bc2[(*wk, *[])]
                self.sum_Bc2[(*wk, *[])] = 1
            self.PP[(*wk, *[])] = self.power_spectrum[(*wk, *[])] * (1 - self.sum_Bc2[(*wk, *[])])

    def _simulate_bsrm_uni(self, phi):
        coeff = np.sqrt((2 ** (
                self.number_of_dimensions + 1)) * self.power_spectrum *
                        self.frequency_length ** self.number_of_dimensions)
        phi_e = np.exp(phi * 1.0j)
        biphase_e = np.exp(self.biphase * 1.0j)
        b = np.sqrt(1 - self.sum_Bc2) * phi_e
        bc = np.sqrt(self.Bc2)

        phi_e = np.einsum('i...->...i', phi_e)
        b = np.einsum('i...->...i', b)

        for i in itertools.product(*self.ranges):
            wk = np.array(i)
            for j in itertools.product(*[range(k) for k in np.ceil((wk + 1) / 2, dtype=np.int32)]):
                wj = np.array(j)
                wi = wk - wj
                b[(*wk, *[])] = b[(*wk, *[])] + bc[(*wi, *wj)] * biphase_e[(*wi, *wj)] * phi_e[(*wi, *[])] * \
                                phi_e[(*wj, *[])]

        b = np.einsum('...i->i...', b)
        phi_e = np.einsum('...i->i...', phi_e)
        b = b * coeff
        b[np.isnan(b)] = 0
        samples = np.fft.fftn(b, self.number_time_intervals)
        samples = samples[:, np.newaxis]
        return np.real(samples)

    def run(self, nsamples, random_state=None):
        """
        Execute the random sampling in the ``MCS`` class.

        The ``run`` method is the function that performs random sampling in the ``MCS`` class. If `nsamples` is
        provided, the ``run`` method is automatically called when the ``MCS`` object is defined. The user may also call
        the ``run`` method directly to generate samples. The ``run`` method of the ``MCS`` class can be invoked many
        times and each time the generated samples are appended to the existing samples.

        ** Input:**

        * **nsamples** (`int`):
            Number of samples to be drawn from each distribution.

            If the ``run`` method is invoked multiple times, the newly generated samples will be appended to the
            existing samples.

        * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
            Random seed used to initialize the pseudo-random number generator. Default is None.

            If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
            object itself can be passed directly.

        **Output/Returns:**

        The ``run`` method has no returns, although it creates and/or appends the `samples` attribute of the ``MCS``
        class.

        """
        # Check if a random_state is provided.

        if nsamples is None:
            raise ValueError('UQpy: Stochastic Process: Number of samples must be defined.')
        if not isinstance(nsamples, int):
            raise ValueError('UQpy: Stochastic Process: nsamples should be an integer.')

        if self.verbose:
            print('UQpy: Stochastic Process: Running 3rd-order Spectral Representation Method.')

        samples = None

        if self.case == 'uni':
            if self.verbose:
                print('UQpy: Stochastic Process: Starting simulation of uni-variate Stochastic Processes.')
                print('UQpy: The number of dimensions is :', self.number_of_dimensions)
            self.phi = np.random.uniform(
                size=np.append(self.nsamples, np.ones(self.number_of_dimensions, dtype=np.int32)
                               * self.number_frequency_intervals)) * 2 * np.pi
            samples = self._simulate_bsrm_uni(self.phi)

        # elif self.case == 'multi':
        #     if self.verbose:
        #         print('UQpy: Stochastic Process: Starting simulation of multi-variate Stochastic Processes.')
        #         print('UQpy: Stochastic Process: The number of variables is :', self.number_of_variables)
        #         print('UQpy: Stochastic Process: The number of dimensions is :', self.number_of_dimensions)
        #     self.phi = np.random.uniform(size=np.append(self.nsamples, np.append(
        #         np.ones(self.number_of_dimensions, dtype=np.int32) * self.number_frequency_intervals,
        #         self.number_of_variables))) * 2 * np.pi
        #     samples = self._simulate_multi(self.phi)

        if self.samples is None:
            self.samples = samples
        else:
            self.samples = np.concatenate((self.samples, samples), axis=0)

        if self.verbose:
            print('UQpy: Stochastic Process: 3rd-order Spectral Representation Method Complete.')


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

        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

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

    :param: dist_object: UQpy Distribution object
    :rtype: dist_object: UQpy.Distribution

    **Output:**

    :param: samples_non_gaussian: Non-Gaussian Stochastic Processes
    :type: samples_non_gaussian: numpy.ndarray

    :param: power_spectrum_non_gaussian: Power Spectrum of the Gaussian Non-Stochastic Processes
    :type: power_spectrum_non_gaussian: numpy.ndarray

    :param: auto_correlation_function_non_gaussian: Auto-covariance Function of the Non-Gaussian Stochastic Processes
    :type: auto_correlation_function_non_gaussian: numpy.ndarray

    **Author:**

    Lohit Vandanapu
    """

    # Created by Lohit Vandanapu
    # Last Modified:04/08/2020 Lohit Vandanapu

    def __init__(self, dist_object, time_duration, frequency_interval, number_time_intervals,
                 number_frequency_intervals, power_spectrum_gaussian=None, auto_correlation_function_gaussian=None,
                 samples_gaussian=None):
        self.dist_object = dist_object
        if auto_correlation_function_gaussian is None and power_spectrum_gaussian is None:
            print('Either the Power Spectrum or the Autocorrelation function should be specified')
        if auto_correlation_function_gaussian is None:
            self.power_spectrum_gaussian = power_spectrum_gaussian
            self.auto_correlation_function_gaussian = S_to_R(power_spectrum_gaussian,
                                                             np.arange(0, number_frequency_intervals) *
                                                             frequency_interval,
                                                             np.arange(0, number_time_intervals) * time_duration)
        elif power_spectrum_gaussian is None:
            self.auto_correlation_function_gaussian = auto_correlation_function_gaussian
            self.power_spectrum_gaussian = R_to_S(auto_correlation_function_gaussian,
                                                  np.arange(0, number_frequency_intervals) * frequency_interval,
                                                  np.arange(0, number_time_intervals) * time_duration)
        self.shape = self.auto_correlation_function_gaussian.shape
        self.dim = len(self.auto_correlation_function_gaussian.shape)
        if samples_gaussian is not None:
            self.samples_gaussian = samples_gaussian
            self.samples_non_gaussian = self.translate_gaussian_samples()
        self.correlation_function_non_gaussian, self.auto_correlation_function_non_gaussian = \
            self.autocorrelation_distortion()
        self.S_ng = R_to_S(self.auto_correlation_function_non_gaussian,
                           np.arange(0, number_frequency_intervals) * frequency_interval,
                           np.arange(0, number_time_intervals) * time_duration)

    def translate_gaussian_samples(self):
        standard_deviation = np.sqrt(self.auto_correlation_function_gaussian[0])
        samples_cdf = norm.cdf(self.samples_gaussian, scale=standard_deviation)
        samples_non_gaussian = None
        if hasattr(self.dist_object, 'icdf'):
            non_gaussian_icdf = getattr(self.dist_object, 'icdf')
            samples_non_gaussian = non_gaussian_icdf(samples_cdf)
        else:
            raise AttributeError('UQpy: The marginal distribution needs to have an inverse cdf defined.')
        return samples_non_gaussian

    def autocorrelation_distortion(self):
        correlation_function_gaussian = R_to_r(self.auto_correlation_function_gaussian)
        correlation_function_gaussian = np.clip(correlation_function_gaussian, -0.999, 0.999)
        correlation_function_non_gaussian = np.zeros_like(correlation_function_gaussian)
        non_gaussian_moments = None
        for i in itertools.product(*[range(s) for s in self.shape]):
            correlation_function_non_gaussian[i] = solve_single_integral(self.dist_object,
                                                                         correlation_function_gaussian[i])
        if hasattr(self.dist_object, 'moments'):
            non_gaussian_moments = getattr(self.dist_object, 'moments')()
        else:
            raise AttributeError('UQpy: The marginal distribution needs to have defined moments.')
        auto_correlation_function_non_gaussian = correlation_function_non_gaussian * non_gaussian_moments[1]
        # TODO: change correlation_function... to auto_correlation_..._standard
        return correlation_function_non_gaussian, auto_correlation_function_non_gaussian


class InverseTranslation:
    """
    A class to perform Iterative Translation Approximation Method to find the underlying  Gaussian Stochastic Processes
    which upon translation would yield the necessary non-Gaussian Stochastic Processes

    **Input:**

    :param: samples_non_gaussian: Non-Gaussian Stochastic Processes
    :type: samples_non_gaussian: numpy.ndarray

    :param: power_spectrum_non_gaussian: Power Spectrum of the Gaussian Non-Stochastic Processes
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

    def __init__(self, distribution, time_duration, frequency_interval, number_time_intervals,
                 number_frequency_intervals, auto_correlation_function_non_gaussian=None,
                 power_spectrum_non_gaussian=None, samples_non_gaussian=None):
        self.distribution = distribution
        self.frequency = np.arange(0, number_frequency_intervals) * frequency_interval
        self.time = np.arange(0, number_time_intervals) * time_duration
        if auto_correlation_function_non_gaussian is None and power_spectrum_non_gaussian is None:
            print('Either the Power Spectrum or the Autocorrelation function should be specified')
        if auto_correlation_function_non_gaussian is None:
            self.power_spectrum_non_gaussian = power_spectrum_non_gaussian
            self.auto_correlation_function_non_gaussian = S_to_R(power_spectrum_non_gaussian, self.frequency, self.time)
        elif power_spectrum_non_gaussian is None:
            self.auto_correlation_function_non_gaussian = auto_correlation_function_non_gaussian
            self.power_spectrum_non_gaussian = R_to_S(auto_correlation_function_non_gaussian, self.frequency, self.time)
        self.num = self.auto_correlation_function_non_gaussian.shape[0]
        self.dim = len(self.auto_correlation_function_non_gaussian.shape)
        if samples_non_gaussian is not None:
            self.samples_non_gaussian = samples_non_gaussian
            self.samples_gaussian = self.inverse_translate_non_gaussian_samples()
        self.power_spectrum_gaussian = self.itam()
        self.auto_correlation_function_gaussian = S_to_R(self.power_spectrum_gaussian, self.frequency, self.time)
        self.correlation_function_gaussian = self.auto_correlation_function_gaussian / \
                                             self.auto_correlation_function_gaussian[0]

    def inverse_translate_non_gaussian_samples(self):
        # TODO: error checks
        samples_cdf = self.distribution.cdf(self.samples_non_gaussian)
        samples_g = Normal(loc=0.0, scale=1.0).icdf(samples_cdf)
        return samples_g

    # TODO: rename to itam_power_spectrum
    def itam(self):
        # Initial Guess
        target_s = self.power_spectrum_non_gaussian
        # Iteration Conditions
        i_converge = 0
        error0 = 100
        max_iter = 10
        target_r = S_to_R(target_s, self.frequency, self.time)
        r_g_iterate = target_r
        s_g_iterate = target_s
        r_ng_iterate = np.zeros_like(target_r)
        s_ng_iterate = np.zeros_like(target_s)

        for ii in range(max_iter):
            r_g_iterate = S_to_R(s_g_iterate, self.frequency, self.time)
            # for i in itertools.product(*[range(self.num) for _ in range(self.dim)]):
            for i in range(len(target_r)):
                r_ng_iterate[i] = solve_single_integral(dist_object=self.distribution,
                                                        rho=r_g_iterate[i] / r_g_iterate[0])
            s_ng_iterate = R_to_S(r_ng_iterate, self.frequency, self.time)

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

        return s_g_iterate

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
    Singular Value Decomposition as opposed to Cholesky Decomposition to be robust with near-Positive Definite
    multi-dimensional Power Spectra.

    **Input:**

    * **nsamples** (`int`):
        Number of samples of the stochastic process to be simulated.

        The ``run`` method is automatically called if `nsamples` is provided. If `nsamples` is not provided, then the
        ``SRM`` object is created but samples are not generated.

    * **power_spectrum** (`list or numpy.ndarray`):
        The prescribed power spectrum.

    * **time_duration** (`list or numpy.ndarray`):
        List of length of time discretizations across dimensions.

        The length of the list needs to be the same as the number of dimensions.

    * **frequency_interval** (`list or numpy.ndarray`):
        List of length of frequency discretizations across dimensions.

        The length of the list needs to be the same as the number of dimensions.

    * **number_frequency_intervals** (`list or numpy.ndarray`):
        List of number of frequency discretizations across dimensions.

        The length of the list needs to be the same as the number of dimensions.

    * **number_time_intervals** (`list or numpy.ndarray`):
        List of number of time discretizations across dimensions.

        The length of the list needs to be the same as the number of dimensions.

    * **case** (`string`):
        The type of simulation - 'uni' for uni-variate and 'multi' for multi-variate.

        The default value of case variable if 'uni'.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (Boolean):
        A boolean declaring whether to write text to the terminal.

    **Attributes:**

    * **samples** (`ndarray`):
        Generated samples.

        The shape of the samples is determined by the number of dimensions and number of variables of the stochastic
        process.

    * **number_of_dimensions** (`int`):
        The dimensionality of the stochastic process.

    * **number_of_variables** (`int`):
        Number of variables in the stochastic process.

    * **phi** (`ndarray`):
        The random phase angles used in the simulation of the stochastic process.

        The shape of the phase angles is determined by the number of dimensions and number of variables of the
        stochastic process.

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
            self.run(nsamples=self.nsamples)

    def run(self, nsamples):
        """
        Execute the random sampling in the ``SRM`` class.

        The ``run`` method is the function that performs random sampling in the ``SRM`` class. If `nsamples` is
        provided, the ``run`` method is automatically called when the ``SRM`` object is defined. The user may also call
        the ``run`` method directly to generate samples. The ``run`` method of the ``SRM`` class can be invoked many
        times and each time the generated samples are appended to the existing samples.

        ** Input:**

        * **nsamples** (`int`):
            Number of samples of the stochastic process to be simulated.

            If the ``run`` method is invoked multiple times, the newly generated samples will be appended to the
            existing samples.

        **Output/Returns:**

        The ``run`` method has no returns, although it creates and/or appends the `samples` attribute of the ``SRM``
        class.

        """

        if nsamples is None:
            raise ValueError('UQpy: Stochastic Process: Number of samples must be defined.')
        if not isinstance(nsamples, int):
            raise ValueError('UQpy: Stochastic Process: nsamples should be an integer.')

        if self.verbose:
            print('UQpy: Stochastic Process: Running Spectral Representation Method.')

        samples = None
        phi = None

        if self.case == 'uni':
            if self.verbose:
                print('UQpy: Stochastic Process: Starting simulation of uni-variate Stochastic Processes.')
                print('UQpy: The number of dimensions is :', self.number_of_dimensions)
            phi = np.random.uniform(
                size=np.append(self.nsamples, np.ones(self.number_of_dimensions, dtype=np.int32)
                               * self.number_frequency_intervals)) * 2 * np.pi
            samples = self._simulate_uni(phi)

        elif self.case == 'multi':
            if self.verbose:
                print('UQpy: Stochastic Process: Starting simulation of multi-variate Stochastic Processes.')
                print('UQpy: Stochastic Process: The number of variables is :', self.number_of_variables)
                print('UQpy: Stochastic Process: The number of dimensions is :', self.number_of_dimensions)
            phi = np.random.uniform(size=np.append(self.nsamples, np.append(
                np.ones(self.number_of_dimensions, dtype=np.int32) * self.number_frequency_intervals,
                self.number_of_variables))) * 2 * np.pi
            samples = self._simulate_multi(phi)

        if self.samples is None:
            self.samples = samples
            self.phi = phi
        else:
            self.samples = np.concatenate((self.samples, samples), axis=0)
            self.phi = np.concatenate((self.phi, phi), axis=0)

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
    A class to simulate Stochastic Processes from a given power spectrum and bispectrum density based on the 3-rd order
    Spectral Representation Method.This class can simulate both uni-variate and multi-variate multi-dimensional
    Stochastic Processes. This class uses Singular Value Decomposition as opposed to Cholesky Decomposition to be robust
    with near-Positive Definite multi-dimensional Power Spectra.

    **Input:**

    * **nsamples** (`int`):
        Number of samples of the stochastic process to be simulated.

        The ``run`` method is automatically called if `nsamples` is provided. If `nsamples` is not provided, then the
        ``BSRM`` object is created but samples are not generated.

    * **power_spectrum** (`list or numpy.ndarray`):
        The prescribed power spectrum.

    * **bispectrum** (`list or numpy.ndarray`):
        The prescribed bispectrum.

    * **time_duration** (`list or numpy.ndarray`):
        List of length of time discretizations across dimensions.

        The length of the list needs to be the same as the number of dimensions.

    * **frequency_interval** (`list or numpy.ndarray`):
        List of length of frequency discretizations across dimensions.

        The length of the list needs to be the same as the number of dimensions.

    * **number_frequency_intervals** (`list or numpy.ndarray`):
        List of number of frequency discretizations across dimensions.

        The length of the list needs to be the same as the number of dimensions.

    * **number_time_intervals** (`list or numpy.ndarray`):
        List of number of time discretizations across dimensions.

        The length of the list needs to be the same as the number of dimensions.

    * **case** (`string`):
        The type of simulation - 'uni' for uni-variate and 'multi' for multi-variate.

        The default value of case variable if 'uni'.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (Boolean):
        A boolean declaring whether to write text to the terminal.

    **Attributes:**

    * **samples** (`ndarray`):
        Generated samples.

        The shape of the samples is determined by the number of dimensions and number of variables of the stochastic
        process.

    * **number_of_dimensions** (`int`):
        The dimensionality of the stochastic process.

    * **number_of_variables** (`int`):
        Number of variables in the stochastic process.

    * **phi** (`ndarray`):
        The random phase angles used in the simulation of the stochastic process.

        The shape of the phase angles is determined by the number of dimensions and number of variables of the
        stochastic process.

    * **b_ampl** (`ndarray`):
        The amplitude of the Bispectral density.

    * **b_real** (`ndarray`):
        The real part of the Bispectral density.

    * **b_imag** (`ndarray`):
        The imaginary part of the Bispectral density.

    * **biphase** (`ndarray`):
        The biphase values of the Bispectral density. THe biphase if defines as the tan inverse of the imaginary part to
        the real part of the bispectral density.

    * **pure_power_spectrum** (`ndarray`):
        The pure part of the power spectrum.

    * **bc2** (`ndarray`):
        The bicoherence values of the power spectrum and bispectrum.

    * **sum_bc2** (`ndarray`):
        The sum of the bicoherence values for single frequencies.

    **Methods**
    """

    def __init__(self, nsamples, power_spectrum, bispectrum, time_duration, frequency_length, number_time_intervals,
                 number_frequency_intervals, case='uni', random_state=None, verbose=False):
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

        self.phi = None
        self.samples = None

        self.case = case
        self.verbose = verbose

        if self.case == 'uni':
            self.number_of_dimensions = len(self.power_spectrum.shape)
            self._compute_bicoherence_uni()
        elif self.case == 'multi':
            self.number_of_variables = self.power_spectrum.shape[0]
            self.number_of_dimensions = len(self.power_spectrum.shape[2:])

        if self.nsamples is not None:
            self.run(nsamples=self.nsamples)

    def _compute_bicoherence_uni(self):
        if self.verbose:
            print('UQpy: Stochastic Process: Computing the partial bicoherence values.')
        self.bc2 = np.zeros_like(self.b_real)
        self.pure_power_sepctrum = np.zeros_like(self.power_spectrum)
        self.sum_bc2 = np.zeros_like(self.power_spectrum)

        if self.number_of_dimensions == 1:
            self.pure_power_sepctrum[0] = self.power_spectrum[0]
            self.pure_power_sepctrum[1] = self.power_spectrum[1]

        if self.number_of_dimensions == 2:
            self.pure_power_sepctrum[0, :] = self.power_spectrum[0, :]
            self.pure_power_sepctrum[1, :] = self.power_spectrum[1, :]
            self.pure_power_sepctrum[:, 0] = self.power_spectrum[:, 0]
            self.pure_power_sepctrum[:, 1] = self.power_spectrum[:, 1]

        if self.number_of_dimensions == 3:
            self.pure_power_sepctrum[0, :, :] = self.power_spectrum[0, :, :]
            self.pure_power_sepctrum[1, :, :] = self.power_spectrum[1, :, :]
            self.pure_power_sepctrum[:, 0, :] = self.power_spectrum[:, 0, :]
            self.pure_power_sepctrum[:, 1, :] = self.power_spectrum[:, 1, :]
            self.pure_power_sepctrum[:, :, 0] = self.power_spectrum[:, :, 0]
            self.pure_power_sepctrum[:, 0, 1] = self.power_spectrum[:, :, 1]

        self.ranges = [range(self.number_frequency_intervals[i]) for i in range(self.number_of_dimensions)]

        for i in itertools.product(*self.ranges):
            wk = np.array(i)
            for j in itertools.product(*[range(np.int32(k)) for k in np.ceil((wk + 1) / 2)]):
                wj = np.array(j)
                wi = wk - wj
                if self.b_ampl[(*wi, *wj)] > 0 and self.pure_power_sepctrum[(*wi, *[])] * \
                        self.pure_power_sepctrum[(*wj, *[])] != 0:
                    self.bc2[(*wi, *wj)] = self.b_ampl[(*wi, *wj)] ** 2 / (
                            self.pure_power_sepctrum[(*wi, *[])] * self.pure_power_sepctrum[(*wj, *[])] *
                            self.power_spectrum[(*wk, *[])]) * self.frequency_length ** self.number_of_dimensions
                    self.sum_bc2[(*wk, *[])] = self.sum_bc2[(*wk, *[])] + self.bc2[(*wi, *wj)]
                else:
                    self.bc2[(*wi, *wj)] = 0
            if self.sum_bc2[(*wk, *[])] > 1:
                print('UQpy: Stochastic Process: Results may not be as expected as sum of partial bicoherences is '
                      'greater than 1')
                for j in itertools.product(*[range(k) for k in np.ceil((wk + 1) / 2, dtype=np.int32)]):
                    wj = np.array(j)
                    wi = wk - wj
                    self.bc2[(*wi, *wj)] = self.bc2[(*wi, *wj)] / self.sum_bc2[(*wk, *[])]
                self.sum_bc2[(*wk, *[])] = 1
            self.pure_power_sepctrum[(*wk, *[])] = self.power_spectrum[(*wk, *[])] * (1 - self.sum_bc2[(*wk, *[])])

    def _simulate_bsrm_uni(self, phi):
        coeff = np.sqrt((2 ** (
                self.number_of_dimensions + 1)) * self.power_spectrum *
                        self.frequency_length ** self.number_of_dimensions)
        phi_e = np.exp(phi * 1.0j)
        biphase_e = np.exp(self.biphase * 1.0j)
        b = np.sqrt(1 - self.sum_bc2) * phi_e
        bc = np.sqrt(self.bc2)

        phi_e = np.einsum('i...->...i', phi_e)
        b = np.einsum('i...->...i', b)

        for i in itertools.product(*self.ranges):
            wk = np.array(i)
            for j in itertools.product(*[range(np.int32(k)) for k in np.ceil((wk + 1) / 2)]):
                wj = np.array(j)
                wi = wk - wj
                b[(*wk, *[])] = b[(*wk, *[])] + bc[(*wi, *wj)] * biphase_e[(*wi, *wj)] * phi_e[(*wi, *[])] * \
                                phi_e[(*wj, *[])]

        b = np.einsum('...i->i...', b)
        b = b * coeff
        b[np.isnan(b)] = 0
        samples = np.fft.fftn(b, self.number_time_intervals)
        samples = samples[:, np.newaxis]
        return np.real(samples)

    def run(self, nsamples):
        """
        Execute the random sampling in the ``BSRM`` class.

        The ``run`` method is the function that performs random sampling in the ``BSRM`` class. If `nsamples` is
        provided, the ``run`` method is automatically called when the ``BSRM`` object is defined. The user may also call
        the ``run`` method directly to generate samples. The ``run`` method of the ``BSRM`` class can be invoked many
        times and each time the generated samples are appended to the existing samples.

        ** Input:**

        * **nsamples** (`int`):
            Number of samples of the stochastic process to be simulated.

            If the ``run`` method is invoked multiple times, the newly generated samples will be appended to the
            existing samples.

        **Output/Returns:**

            The ``run`` method has no returns, although it creates and/or appends the `samples` attribute of the
            ``BSRM`` class.

        """

        if nsamples is None:
            raise ValueError('UQpy: Stochastic Process: Number of samples must be defined.')
        if not isinstance(nsamples, int):
            raise ValueError('UQpy: Stochastic Process: nsamples should be an integer.')

        if self.verbose:
            print('UQpy: Stochastic Process: Running 3rd-order Spectral Representation Method.')

        samples = None
        phi = None

        if self.case == 'uni':
            if self.verbose:
                print('UQpy: Stochastic Process: Starting simulation of uni-variate Stochastic Processes.')
                print('UQpy: The number of dimensions is :', self.number_of_dimensions)
            phi = np.random.uniform(
                size=np.append(self.nsamples, np.ones(self.number_of_dimensions, dtype=np.int32)
                               * self.number_frequency_intervals)) * 2 * np.pi
            samples = self._simulate_bsrm_uni(phi)

        if self.samples is None:
            self.samples = samples
            self.phi = phi
        else:
            self.samples = np.concatenate((self.samples, samples), axis=0)
            self.phi = np.concatenate((self.phi, phi), axis=0)

        if self.verbose:
            print('UQpy: Stochastic Process: 3rd-order Spectral Representation Method Complete.')


class KLE:
    """
    A class to simulate Stochastic Processes from a given auto-correlation function based on the Karhunen-Louve
    Expansion

    **Input:**

    * **nsamples** (`int`):
        Number of samples of the stochastic process to be simulated.

        The ``run`` method is automatically called if `nsamples` is provided. If `nsamples` is not provided, then the
        ``KLE`` object is created but samples are not generated.

    * **correlation_function** (`list or numpy.ndarray`):
        The correlation function of the stochastic process.

    * **time_duration** (`float`):
        The length of time discretization.

    * **threashold** (`int`):
        The threshold of number of eigen values to be used in the expansion.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (Boolean):
        A boolean declaring whether to write text to the terminal.

    **Attributes:**

    * **samples** (`ndarray`):
        Array of generated samples.

    * **xi** (`ndarray`):
        The independent gaussian random variables used in the expansion.

    **Methods**
    """

    def __init__(self, nsamples, correlation_function, time_duration, threshold=None, random_state=None, verbose=False):
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

        self.verbose = verbose
        self.nsamples = nsamples

        self.samples = None
        self.xi = None

        if self.nsamples is not None:
            self.run(nsamples=self.nsamples)

    def _simulate(self, xi):
        lam, phi = np.linalg.eig(self.correlation_function)
        lam = np.diag(lam)
        lam = lam.astype(np.float64)
        samples = np.dot(phi[:, :self.number_eigen_values], np.dot(sqrtm(lam[:self.number_eigen_values]), xi))
        samples = np.real(samples)
        samples = samples.T
        samples = samples[:, np.newaxis]
        return samples

    def run(self, nsamples):
        """
        Execute the random sampling in the ``KLE`` class.

        The ``run`` method is the function that performs random sampling in the ``KLE`` class. If `nsamples` is
        provided, the ``run`` method is automatically called when the ``KLE`` object is defined. The user may also call
        the ``run`` method directly to generate samples. The ``run`` method of the ``KLE`` class can be invoked many
        times and each time the generated samples are appended to the existing samples.

        ** Input:**

        * **nsamples** (`int`):
            Number of samples of the stochastic process to be simulated.

            If the ``run`` method is invoked multiple times, the newly generated samples will be appended to the
            existing samples.

        **Output/Returns:**

            The ``run`` method has no returns, although it creates and/or appends the `samples` attribute of the
            ``KLE`` class.

        """

        if nsamples is None:
            raise ValueError('UQpy: Stochastic Process: Number of samples must be defined.')
        if not isinstance(nsamples, int):
            raise ValueError('UQpy: Stochastic Process: nsamples should be an integer.')

        if self.verbose:
            print('UQpy: Stochastic Process: Running Karhunen Loeve Expansion.')

        if self.verbose:
            print('UQpy: Stochastic Process: Starting simulation of Stochastic Processes.')
        xi = np.random.normal(size=(self.number_eigen_values, self.nsamples))
        samples = self._simulate(xi)

        if self.samples is None:
            self.samples = samples
            self.xi = xi
        else:
            self.samples = np.concatenate((self.samples, samples), axis=0)
            self.xi = np.concatenate((self.xi, xi), axis=0)

        if self.verbose:
            print('UQpy: Stochastic Process: Karhunen-Loeve Expansion Complete.')


class Translation:
    """
    A class to translate Gaussian Stochastic Processes to non-Gaussian Stochastic Processes

    **Input:**

    * **dist_object** (`list or numpy.ndarray`):
        An instance of the UQpy Distributions class defining the marginal distribution to which the gaussian stochastic
        process should be translated to.

    * **time_duration** (`float`):
        The value of time discretization.

    * **frequency_interval** (`float`):
        The value of frequency discretization.

    * **number_time_intervals** (`int`):
        The number of time discretizations.

    * **number_frequency_intervals** (`int`):
        The number of frequency discretizations.

    * **power_spectrum_gaussian** ('list or numpy.ndarray'):
        The power spectrum of the gaussian stochastic process to be translated.

    * **correlation_function_gaussian** ('list or numpy.ndarray'):
        The auto correlation function of the gaussian stochastic process to be translated.

        Either the power spectrum or the auto correlation function of the gaussian stochastic process needs to be
        defined.

    * **samples_gaussian** (`list or numpy.ndarray`):
        Samples of gaussian stochastic process to be translated.

    **Attributes:**

    * **samples_non_gaussian** (`numpy.ndarray`):
        Translated non-gaussian stochastic process from gaussian stochastic processes.

    * **power_spectrum_non_gaussian** (`numpy.ndarray`):
        The power spectrum of the translated non-gaussian stochastic processes.

    * **correlation_function_non_gaussian** (`numpy.ndarray`):
        The correlation function of the translated non-gaussian stochastic processes obtained by distorting the gaussian
        correlation function.

    * **scaled_correlation_function_non_gaussian** (`numpy.ndarray`):
        This obtained by scaling the correlation function of the non-gaussian stochastic processes to make the
        correlation at '0' distance to be 1

    **Methods**
    """

    def __init__(self, dist_object, time_duration, frequency_interval, number_time_intervals,
                 number_frequency_intervals, power_spectrum_gaussian=None, correlation_function_gaussian=None,
                 samples_gaussian=None):
        self.dist_object = dist_object
        if correlation_function_gaussian is None and power_spectrum_gaussian is None:
            print('Either the Power Spectrum or the Autocorrelation function should be specified')
        if correlation_function_gaussian is None:
            self.power_spectrum_gaussian = power_spectrum_gaussian
            self.correlation_function_gaussian = S_to_R(power_spectrum_gaussian,
                                                        np.arange(0, number_frequency_intervals) *
                                                        frequency_interval,
                                                        np.arange(0, number_time_intervals) * time_duration)
        elif power_spectrum_gaussian is None:
            self.correlation_function_gaussian = correlation_function_gaussian
            self.power_spectrum_gaussian = R_to_S(correlation_function_gaussian,
                                                  np.arange(0, number_frequency_intervals) * frequency_interval,
                                                  np.arange(0, number_time_intervals) * time_duration)
        self.shape = self.correlation_function_gaussian.shape
        self.dim = len(self.correlation_function_gaussian.shape)
        if samples_gaussian is not None:
            self.samples_gaussian = samples_gaussian
            self.samples_non_gaussian = self.translate_gaussian_samples()
        self.correlation_function_non_gaussian, self.scaled_correlation_function_non_gaussian = \
            self.autocorrelation_distortion()
        self.power_spectrum_non_gaussian = R_to_S(self.correlation_function_non_gaussian,
                           np.arange(0, number_frequency_intervals) * frequency_interval,
                           np.arange(0, number_time_intervals) * time_duration)

    def translate_gaussian_samples(self):
        standard_deviation = np.sqrt(self.correlation_function_gaussian[0])
        samples_cdf = norm.cdf(self.samples_gaussian, scale=standard_deviation)
        if hasattr(self.dist_object, 'icdf'):
            non_gaussian_icdf = getattr(self.dist_object, 'icdf')
            samples_non_gaussian = non_gaussian_icdf(samples_cdf)
        else:
            raise AttributeError('UQpy: The marginal dist_object needs to have an inverse cdf defined.')
        return samples_non_gaussian

    def autocorrelation_distortion(self):
        correlation_function_gaussian = R_to_r(self.correlation_function_gaussian)
        correlation_function_gaussian = np.clip(correlation_function_gaussian, -0.999, 0.999)
        correlation_function_non_gaussian = np.zeros_like(correlation_function_gaussian)
        for i in itertools.product(*[range(s) for s in self.shape]):
            correlation_function_non_gaussian[i] = solve_single_integral(self.dist_object,
                                                                         correlation_function_gaussian[i])
        if hasattr(self.dist_object, 'moments'):
            non_gaussian_moments = getattr(self.dist_object, 'moments')()
        else:
            raise AttributeError('UQpy: The marginal dist_object needs to have defined moments.')
        scaled_correlation_function_non_gaussian = correlation_function_non_gaussian * non_gaussian_moments[1]
        return correlation_function_non_gaussian, scaled_correlation_function_non_gaussian


class InverseTranslation:
    """
    A class to perform Iterative Translation Approximation Method to find the underlying  Gaussian Stochastic Processes
    which upon translation would yield the necessary non-Gaussian Stochastic Processes.

    **Input:**

    * **dist_object** (`list or numpy.ndarray`):
        An instance of the UQpy Distributions class defining the marginal distribution of the non-gaussian stochastic
        processes.

    * **time_duration** (`float`):
        The value of time discretization.

    * **frequency_interval** (`float`):
        The value of frequency discretization.

    * **number_time_intervals** (`int`):
        The number of time discretizations.

    * **number_frequency_intervals** (`int`):
        The number of frequency discretizations.

    * **power_spectrum_non_gaussian** ('list or numpy.ndarray'):
        The power spectrum of the non-gaussian stochastic processes.

    * **correlation_function_non_gaussian** ('list or numpy.ndarray'):
        The auto correlation function of the non-gaussian stochastic processes.

        Either the power spectrum or the auto correlation function of the gaussian stochastic process needs to be
        defined.

    * **samples_non_gaussian** (`list or numpy.ndarray`):
        Samples of non-gaussian stochastic processes.

    **Attributes:**

    * **samples_gaussian** (`numpy.ndarray`):
        The Inverse translated gaussian stochastic processes from the non-gaussian stochastic processes.

    * **power_spectrum_gaussian** (`numpy.ndarray`):
        The power spectrum of the inverse translated gaussian stochastic processes.

    * **correlation_function_gaussian** (`numpy.ndarray`):
        The correlation function of the inverse translated gaussian stochastic processes.

    * **scaled_correlation_function_non_gaussian** (`numpy.ndarray`):
        This obtained by scaling the correlation function of the gaussian stochastic processes to make the correlation
        at '0' distance to be 1

    **Methods**
    """

    def __init__(self, dist_object, time_duration, frequency_interval, number_time_intervals,
                 number_frequency_intervals, correlation_function_non_gaussian=None,
                 power_spectrum_non_gaussian=None, samples_non_gaussian=None):
        self.dist_object = dist_object
        self.frequency = np.arange(0, number_frequency_intervals) * frequency_interval
        self.time = np.arange(0, number_time_intervals) * time_duration
        if correlation_function_non_gaussian is None and power_spectrum_non_gaussian is None:
            print('Either the Power Spectrum or the Autocorrelation function should be specified')
        if correlation_function_non_gaussian is None:
            self.power_spectrum_non_gaussian = power_spectrum_non_gaussian
            self.correlation_function_non_gaussian = S_to_R(power_spectrum_non_gaussian, self.frequency, self.time)
        elif power_spectrum_non_gaussian is None:
            self.correlation_function_non_gaussian = correlation_function_non_gaussian
            self.power_spectrum_non_gaussian = R_to_S(correlation_function_non_gaussian, self.frequency, self.time)
        self.num = self.correlation_function_non_gaussian.shape[0]
        self.dim = len(self.correlation_function_non_gaussian.shape)
        if samples_non_gaussian is not None:
            self.samples_non_gaussian = samples_non_gaussian
            self.samples_gaussian = self.inverse_translate_non_gaussian_samples()
        self.power_spectrum_gaussian = self.itam_power_spectrum()
        self.auto_correlation_function_gaussian = S_to_R(self.power_spectrum_gaussian, self.frequency, self.time)
        self.correlation_function_gaussian = self.auto_correlation_function_gaussian / \
                                             self.auto_correlation_function_gaussian[0]

    def inverse_translate_non_gaussian_samples(self):
        if hasattr(self.dist_object, 'cdf'):
            non_gaussian_cdf = getattr(self.dist_object, 'cdf')
            samples_cdf = non_gaussian_cdf(self.samples_non_gaussian)
        else:
            raise AttributeError('UQpy: The marginal dist_object needs to have an inverse cdf defined.')
        samples_g = Normal(loc=0.0, scale=1.0).icdf(samples_cdf)
        return samples_g

    def itam_power_spectrum(self):
        target_s = self.power_spectrum_non_gaussian
        i_converge = 0
        max_iter = 100
        target_r = S_to_R(target_s, self.frequency, self.time)
        r_g_iterate = target_r
        s_g_iterate = target_s
        r_ng_iterate = np.zeros_like(target_r)
        s_ng_iterate = np.zeros_like(target_s)

        for _ in range(max_iter):
            r_g_iterate = S_to_R(s_g_iterate, self.frequency, self.time)
            for i in range(len(target_r)):
                r_ng_iterate[i] = solve_single_integral(dist_object=self.dist_object,
                                                        rho=r_g_iterate[i] / r_g_iterate[0])
            s_ng_iterate = R_to_S(r_ng_iterate, self.frequency, self.time)

            err1 = np.sum((target_s - s_ng_iterate) ** 2)
            err2 = np.sum(target_s ** 2)

            if 100 * np.sqrt(err1 / err2) < 0.0005:
                i_converge = 1

            s_g_next_iterate = (target_s / s_ng_iterate) * s_g_iterate

            # Eliminate Numerical error of Upgrading Scheme
            s_g_next_iterate[s_g_next_iterate < 0] = 0
            s_g_iterate = s_g_next_iterate

            if i_converge:
                break

        return s_g_iterate

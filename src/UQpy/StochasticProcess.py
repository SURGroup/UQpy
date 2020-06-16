"""
The module currently contains the following classes:

* ``SRM``: Class for simulation of Gaussian stochastic processes and random fields using the Spectral Representation
  Method.
* ``BSRM``: Class for simulation of third-order non-Gaussian stochastic processes and random fields using the
  Bispectral Representation Method.
* ``KLE``: Class for simulation of stochastic processes using the Karhunen-Loeve Expansion.
* ``Translation``: Class for transforming a Gaussian stochastic process to a non-Gaussian stochastic process with
  prescribed marginal probability distribution.
* ``InverseTranslation``: Call for identifying an underlying Gaussian stochastic process for a non-Gaussian process with
  prescribed marginal probability distribution and autocorrelation function / power spectrum.
"""

import itertools

from scipy.linalg import sqrtm

from UQpy.Distributions import *
from UQpy.Utilities import *


# TODO: add non-stationary-methods for all the classes


class SRM:
    """
    A class to simulate stochastic processes from a given power spectrum density using the Spectral Representation
    Method. This class can simulate uni-variate, multi-variate, and multi-dimensional stochastic processes. The class
    uses Singular Value Decomposition, as opposed to Cholesky Decomposition, to ensure robust, near-positive definite
    multi-dimensional power spectra.

    **Input:**

    * **nsamples** (`int`):
        Number of samples of the stochastic process to be simulated.

        The ``run`` method is automatically called if `nsamples` is provided. If `nsamples` is not provided, then the
        ``SRM`` object is created but samples are not generated.

    * **power_spectrum** (`list or numpy.ndarray`):
        The discretized power spectrum.

        For uni-variate, one-dimensional processes `power_spectrum` will be `list` or `ndarray` of length
        `number_frequency_intervals`.

        For multi-variate, one-dimensional processes, `power_spectrum` will be a `list` or `ndarray` of size
        (`number_of_variables`, `number_of_variables`, `number_frequency_intervals`).

        For uni-variate, multi-dimensional processes, `power_spectrum` will be a `list` or `ndarray` of size
        (`number_frequency_intervals[0]`, ..., `number_frequency_intervals[number_of_dimensions-1]`)

        For multi-variate, multi-dimensional processes, `power_spectrum` will be a `list` or `ndarray` of size
        (`number_of_variables`, `number_of_variables`, `number_frequency_intervals[0]`, ...
        `number_frequency_intervals[number_of_dimensions-1]``).

    * **time_interval** (`list or numpy.ndarray`):
        Length of time discretizations (:math:`\Delta t`) for each dimension of size `number_of_dimensions`.

    * **frequency_interval** (`list or numpy.ndarray`):
        Length of frequency discretizations (:math:`\Delta \omega`) for each dimension of size `number_of_dimensions`.

    * **number_frequency_intervals** (`list or numpy.ndarray`):
        Number of frequency discretizations for each dimension of size `number_of_dimensions`.

    * **number_time_intervals** (`list or numpy.ndarray`):
        Number of time discretizations for each dimensions of size `number_of_dimensions`.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (Boolean):
        A boolean declaring whether to write text to the terminal.

    **Attributes:**

    * **samples** (`ndarray`):
        Generated samples.

        The shape of the samples is (`nsamples`, `number_of_variables`, `number_time_intervals[0]`, ...,
        `number_time_intervals[number_of_dimensions-1]`)

    * **number_of_dimensions** (`int`):
        The dimensionality of the stochastic process.

    * **number_of_variables** (`int`):
        Number of variables in the stochastic process.

    * **phi** (`ndarray`):
        The random phase angles used in the simulation of the stochastic process.

        The shape of the phase angles (`nsamples`, `number_of_variables`, `number_frequency_intervals[0]`, ...,
        `number_frequency_intervals[number_of_dimensions-1]`)

    **Methods**

    """

    def __init__(self, nsamples, power_spectrum, time_interval, frequency_interval, number_time_intervals,
                 number_frequency_intervals, random_state=None, verbose=False):
        self.power_spectrum = power_spectrum
        if isinstance(time_interval, float) and isinstance(frequency_interval, float) and \
                isinstance(number_time_intervals, int) and isinstance(number_frequency_intervals, int):
            time_interval = [time_interval]
            frequency_interval = [frequency_interval]
            number_time_intervals = [number_time_intervals]
            number_frequency_intervals = [number_frequency_intervals]
        self.time_interval = np.array(time_interval)
        self.frequency_interval = np.array(frequency_interval)
        self.number_time_intervals = np.array(number_time_intervals)
        self.number_frequency_intervals = np.array(number_frequency_intervals)
        self.nsamples = nsamples

        # Error checks
        t_u = 2 * np.pi / (2 * self.number_frequency_intervals * self.frequency_interval)
        if (self.time_interval > t_u).any():
            raise RuntimeError('UQpy: Aliasing might occur during execution')

        self.verbose = verbose

        self.random_state = random_state
        if isinstance(self.random_state, int):
            np.random.seed(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        self.samples = None
        self.number_of_variables = None
        self.number_of_dimensions = len(self.number_frequency_intervals)
        self.phi = None

        if self.number_of_dimensions == len(self.power_spectrum.shape):
            self.case = 'uni'
        else:
            self.number_of_variables = self.power_spectrum.shape[0]
            self.case = 'multi'

        # Run Spectral Representation Method
        if self.nsamples is not None:
            self.run(nsamples=self.nsamples)

    def run(self, nsamples):
        """
        Execute the random sampling in the ``SRM`` class.

        The ``run`` method is the function that performs random sampling in the ``SRM`` class. If `nsamples` is
        provided when the ``SRM`` object is defined, the ``run`` method is automatically called. The user may also call
        the ``run`` method directly to generate samples. The ``run`` method of the ``SRM`` class can be invoked many
        times and each time the generated samples are appended to the existing samples.

        **Input:**

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
            2 ** (self.number_of_dimensions + 1) * self.power_spectrum * np.prod(self.frequency_interval))
        samples = np.fft.fftn(fourier_coefficient, self.number_time_intervals)
        samples = np.real(samples)
        samples = samples[:, np.newaxis]
        return samples

    def _simulate_multi(self, phi):
        power_spectrum = np.einsum('ij...->...ij', self.power_spectrum)
        coefficient = np.sqrt(2 ** (self.number_of_dimensions + 1)) * np.sqrt(np.prod(self.frequency_interval))
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
    A class to simulate non-Gaussian stochastic processes from a given power spectrum and bispectrum based on the 3-rd
    order Spectral Representation Method. This class can simulate uni-variate, one-dimensional and multi-dimensional
    stochastic processes.

    **Input:**

    * **nsamples** (`int`):
        Number of samples of the stochastic process to be simulated.

        The ``run`` method is automatically called if `nsamples` is provided. If `nsamples` is not provided, then the
        ``BSRM`` object is created but samples are not generated.

    * **power_spectrum** (`list or numpy.ndarray`):
        The discretized power spectrum.

        For uni-variate, one-dimensional processes `power_spectrum` will be `list` or `ndarray` of length
        `number_frequency_intervals`.

        For uni-variate, multi-dimensional processes, `power_spectrum` will be a `list` or `ndarray` of size
        (`number_frequency_intervals[0]`, ..., `number_frequency_intervals[number_of_dimensions-1]`)

    * **bispectrum** (`list or numpy.ndarray`):
        The prescribed bispectrum.

        For uni-variate, one-dimensional processes, `bispectrum` will be a `list` or `ndarray` of size
        (`number_frequency_intervals`, `number_frequency_intervals`)

        For uni-variate, multi-dimensional processes, `bispectrum` will be a `list` or `ndarray` of size
        (`number_frequency_intervals[0]`, ..., `number_frequency_intervals[number_of_dimensions-1]`,
        `number_frequency_intervals[0]`, ..., `number_frequency_intervals[number_of_dimensions-1]`)

    * **time_interval** (`list or numpy.ndarray`):
        Length of time discretizations (:math:`\Delta t`) for each dimension of size `number_of_dimensions`.

    * **frequency_interval** (`list or numpy.ndarray`):
        Length of frequency discretizations (:math:`\Delta \omega`) for each dimension of size `number_of_dimensions`.

    * **number_frequency_intervals** (`list or numpy.ndarray`):
        Number of frequency discretizations for each dimension of size `number_of_dimensions`.

    * **number_time_intervals** (`list or numpy.ndarray`):
        Number of time discretizations for each dimensions of size `number_of_dimensions`.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **verbose** (Boolean):
        A boolean declaring whether to write text to the terminal.

    **Attributes:**

    * **samples** (`ndarray`):
        Generated samples.

        The shape of the samples is (`nsamples`, `number_of_variables`, `number_time_intervals[0]`, ...,
        `number_time_intervals[number_of_dimensions-1]`)

    * **number_of_dimensions** (`int`):
        The dimensionality of the stochastic process.

    * **number_of_variables** (`int`):
        Number of variables in the stochastic process.

    * **phi** (`ndarray`):
        The random phase angles used in the simulation of the stochastic process.

        The shape of the phase angles (`nsamples`, `number_of_variables`, `number_frequency_intervals[0]`, ...,
        `number_frequency_intervals[number_of_dimensions-1]`)

    * **b_ampl** (`ndarray`):
        The amplitude of the bispectrum.

    * **b_real** (`ndarray`):
        The real part of the bispectrum.

    * **b_imag** (`ndarray`):
        The imaginary part of the bispectrum.

    * **biphase** (`ndarray`):
        The biphase values of the bispectrum.

    * **pure_power_spectrum** (`ndarray`):
        The pure part of the power spectrum.

    * **bc2** (`ndarray`):
        The bicoherence values of the power spectrum and bispectrum.

    * **sum_bc2** (`ndarray`):
        The sum of the bicoherence values for single frequencies.

    **Methods**
    """

    def __init__(self, nsamples, power_spectrum, bispectrum, time_interval, frequency_interval, number_time_intervals,
                 number_frequency_intervals, case='uni', random_state=None, verbose=False):
        self.nsamples = nsamples
        self.number_frequency_intervals = np.array(number_frequency_intervals)
        self.number_time_intervals = np.array(number_time_intervals)
        self.frequency_interval = np.array(frequency_interval)
        self.time_interval = np.array(time_interval)
        self.number_of_dimensions = len(power_spectrum.shape)
        self.power_spectrum = power_spectrum
        self.bispectrum = bispectrum

        # Error checks
        t_u = 2 * np.pi / (2 * self.number_frequency_intervals * self.frequency_interval)
        if (self.time_interval > t_u).any():
            raise RuntimeError('UQpy: Aliasing might occur during execution')

        self.random_state = random_state
        if isinstance(self.random_state, int):
            np.random.seed(self.random_state)
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

        if self.number_of_dimensions == len(self.power_spectrum.shape):
            self.case = 'uni'
        else:
            self.number_of_variables = self.power_spectrum.shape[0]
            self.case = 'multi'

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
                            self.power_spectrum[(*wk, *[])]) * self.frequency_interval ** self.number_of_dimensions
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
                        self.frequency_interval ** self.number_of_dimensions)
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
    A class to simulate stochastic processes from a given auto-correlation function based on the Karhunen-Loeve
    Expansion

    **Input:**

    * **nsamples** (`int`):
        Number of samples of the stochastic process to be simulated.

        The ``run`` method is automatically called if `nsamples` is provided. If `nsamples` is not provided, then the
        ``KLE`` object is created but samples are not generated.

    * **correlation_function** (`list or numpy.ndarray`):
        The correlation function of the stochastic process of size (`number_time_intervals`, `number_time_intervals`)

    * **time_interval** (`float`):
        The length of time discretization.

    * **threshold** (`int`):
        The threshold number of eigenvalues to be used in the expansion.

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

    # TODO: Test this for non-stationary processes.

    def __init__(self, nsamples, correlation_function, time_interval, threshold=None, random_state=None, verbose=False):
        self.correlation_function = correlation_function
        self.time_interval = time_interval
        if threshold:
            self.number_eigen_values = threshold
        else:
            self.number_eigen_values = len(self.correlation_function[0])

        self.random_state = random_state
        if isinstance(self.random_state, int):
            np.random.seed(self.random_state)
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
        provided when the ``KLE`` object is defined, the ``run`` method is automatically called. The user may also call
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
        An instance of the UQpy ``Distributions`` class defining the marginal distribution to which the Gaussian
        stochastic process should be translated to.

    * **time_interval** (`float`):
        The value of time discretization.

    * **frequency_interval** (`float`):
        The value of frequency discretization.

    * **number_time_intervals** (`int`):
        The number of time discretizations.

    * **number_frequency_intervals** (`int`):
        The number of frequency discretizations.

    * **power_spectrum_gaussian** ('list or numpy.ndarray'):
        The power spectrum of the gaussian stochastic process to be translated.

        `power_spectrum_gaussian` must be of size (`number_frequency_intervals`).

    * **correlation_function_gaussian** ('list or numpy.ndarray'):
        The auto correlation function of the Gaussian stochastic process to be translated.

        Either the power spectrum or the auto correlation function of the gaussian stochastic process needs to be
        defined.

        `correlation_function_gaussian` must be of size (`number_time_intervals`).

    * **samples_gaussian** (`list or numpy.ndarray`):
        Samples of Gaussian stochastic process to be translated.

        `samples_gaussian` is optional. If no samples are passed, the ``Translation`` class will compute the correlation
        distortion.

    **Attributes:**

    * **samples_non_gaussian** (`numpy.ndarray`):
        Translated non-Gaussian stochastic process from Gaussian samples.

    * **power_spectrum_non_gaussian** (`numpy.ndarray`):
        The power spectrum of the translated non-Gaussian stochastic processes.

    * **correlation_function_non_gaussian** (`numpy.ndarray`):
        The correlation function of the translated non-Gaussian stochastic processes obtained by distorting the Gaussian
        correlation function.

    * **scaled_correlation_function_non_gaussian** (`numpy.ndarray`):
        This obtained by scaling the correlation function of the non-Gaussian stochastic processes to make the
        correlation at '0' lag to be 1
    """

    def __init__(self, dist_object, time_interval, frequency_interval, number_time_intervals,
                 number_frequency_intervals, power_spectrum_gaussian=None, correlation_function_gaussian=None,
                 samples_gaussian=None):
        self.dist_object = dist_object
        self.time_interval = time_interval
        self.frequency_interval = frequency_interval
        self.number_time_intervals = number_time_intervals
        self.number_frequency_intervals = number_frequency_intervals
        if correlation_function_gaussian is None and power_spectrum_gaussian is None:
            print('Either the Power Spectrum or the Autocorrelation function should be specified')
        if correlation_function_gaussian is None:
            self.power_spectrum_gaussian = power_spectrum_gaussian
            self.correlation_function_gaussian = wiener_khinchin_transform(power_spectrum_gaussian, np.arange(0,
                                                                           self.number_frequency_intervals) *
                                                                           self.frequency_interval,
                                                                           np.arange(0, self.number_time_intervals) *
                                                                           self.time_interval)
        elif power_spectrum_gaussian is None:
            self.correlation_function_gaussian = correlation_function_gaussian
            self.power_spectrum_gaussian = inverse_wiener_khinchin_transform(correlation_function_gaussian, np.arange(0,
                                                                             self.number_frequency_intervals) *
                                                                             self.frequency_interval,
                                                                             np.arange(0, self.number_time_intervals) *
                                                                             self.time_interval)
        self.shape = self.correlation_function_gaussian.shape
        self.dim = len(self.correlation_function_gaussian.shape)
        if samples_gaussian is not None:
            self.samples_shape = samples_gaussian.shape
            self.samples_gaussian = samples_gaussian.flatten()[:, np.newaxis]
            self.samples_non_gaussian = self._translate_gaussian_samples().reshape(self.samples_shape)
        self.correlation_function_non_gaussian, self.scaled_correlation_function_non_gaussian = \
            self._autocorrelation_distortion()
        self.power_spectrum_non_gaussian = inverse_wiener_khinchin_transform(self.correlation_function_non_gaussian,
                                                                             np.arange(0,
                                                                                       self.number_frequency_intervals)
                                                                             * self.frequency_interval,
                                                                             np.arange(0,
                                                                                       self.number_time_intervals)
                                                                             * self.time_interval)

    def _translate_gaussian_samples(self):
        standard_deviation = np.sqrt(self.correlation_function_gaussian[0])
        samples_cdf = norm.cdf(self.samples_gaussian, scale=standard_deviation)
        if hasattr(self.dist_object, 'icdf'):
            non_gaussian_icdf = getattr(self.dist_object, 'icdf')
            samples_non_gaussian = non_gaussian_icdf(samples_cdf)
        else:
            raise AttributeError('UQpy: The marginal dist_object needs to have an inverse cdf defined.')
        return samples_non_gaussian

    def _autocorrelation_distortion(self):
        correlation_function_gaussian = scaling_correlation_function(self.correlation_function_gaussian)
        correlation_function_gaussian = np.clip(correlation_function_gaussian, -0.999, 0.999)
        correlation_function_non_gaussian = np.zeros_like(correlation_function_gaussian)
        for i in itertools.product(*[range(s) for s in self.shape]):
            correlation_function_non_gaussian[i] = correlation_distortion(self.dist_object,
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
        An instance of the ``UQpy`` ``Distributions`` class defining the marginal distribution of the non-Gaussian
        stochastic process.

    * **time_interval** (`float`):
        The value of time discretization.

    * **frequency_interval** (`float`):
        The value of frequency discretization.

    * **number_time_intervals** (`int`):
        The number of time discretizations.

    * **number_frequency_intervals** (`int`):
        The number of frequency discretizations.

    * **power_spectrum_non_gaussian** ('list or numpy.ndarray'):
        The power spectrum of the non-Gaussian stochastic processes.

    * **correlation_function_non_gaussian** ('list or numpy.ndarray'):
        The auto correlation function of the non-Gaussian stochastic processes.

        Either the power spectrum or the auto correlation function of the Gaussian stochastic process needs to be
        defined.

    * **samples_non_gaussian** (`list or numpy.ndarray`):
        Samples of non-Gaussian stochastic processes.

        `samples_non_gaussian` is optional. If no samples are passed, the ``InverseTranslation`` class will compute the
        underlying Gaussian correlation using the ITAM.

    **Attributes:**

    * **samples_gaussian** (`numpy.ndarray`):
        The inverse translated Gaussian samples from the non-Gaussian samples.

    * **power_spectrum_gaussian** (`numpy.ndarray`):
        The power spectrum of the inverse translated Gaussian stochastic processes.

    * **correlation_function_gaussian** (`numpy.ndarray`):
        The correlation function of the inverse translated Gaussian stochastic processes.

    * **scaled_correlation_function_non_gaussian** (`numpy.ndarray`):
        This obtained by scaling the correlation function of the Gaussian stochastic processes to make the correlation
        at '0' distance to be 1

    """

    def __init__(self, dist_object, time_interval, frequency_interval, number_time_intervals,
                 number_frequency_intervals, correlation_function_non_gaussian=None,
                 power_spectrum_non_gaussian=None, samples_non_gaussian=None):
        self.dist_object = dist_object
        self.frequency = np.arange(0, number_frequency_intervals) * frequency_interval
        self.time = np.arange(0, number_time_intervals) * time_interval
        if correlation_function_non_gaussian is None and power_spectrum_non_gaussian is None:
            print('Either the Power Spectrum or the Autocorrelation function should be specified')
        if correlation_function_non_gaussian is None:
            self.power_spectrum_non_gaussian = power_spectrum_non_gaussian
            self.correlation_function_non_gaussian = wiener_khinchin_transform(power_spectrum_non_gaussian,
                                                                               self.frequency, self.time)
        elif power_spectrum_non_gaussian is None:
            self.correlation_function_non_gaussian = correlation_function_non_gaussian
            self.power_spectrum_non_gaussian = inverse_wiener_khinchin_transform(correlation_function_non_gaussian,
                                                                                 self.frequency, self.time)
        self.num = self.correlation_function_non_gaussian.shape[0]
        self.dim = len(self.correlation_function_non_gaussian.shape)
        if samples_non_gaussian is not None:
            self.samples_shape = samples_non_gaussian.shape
            self.samples_non_gaussian = samples_non_gaussian.flatten()[:, np.newaxis]
            self.samples_gaussian = self._inverse_translate_non_gaussian_samples().reshape(self.samples_shape)
        self.power_spectrum_gaussian = self._itam_power_spectrum()
        self.auto_correlation_function_gaussian = wiener_khinchin_transform(self.power_spectrum_gaussian,
                                                                            self.frequency, self.time)
        self.correlation_function_gaussian = self.auto_correlation_function_gaussian / \
                                             self.auto_correlation_function_gaussian[0]

    def _inverse_translate_non_gaussian_samples(self):
        if hasattr(self.dist_object, 'cdf'):
            non_gaussian_cdf = getattr(self.dist_object, 'cdf')
            samples_cdf = non_gaussian_cdf(self.samples_non_gaussian)
        else:
            raise AttributeError('UQpy: The marginal dist_object needs to have an inverse cdf defined.')
        samples_g = Normal(loc=0.0, scale=1.0).icdf(samples_cdf)
        return samples_g

    def _itam_power_spectrum(self):
        target_s = self.power_spectrum_non_gaussian
        i_converge = 0
        max_iter = 100
        target_r = wiener_khinchin_transform(target_s, self.frequency, self.time)
        r_g_iterate = target_r
        s_g_iterate = target_s
        r_ng_iterate = np.zeros_like(target_r)
        s_ng_iterate = np.zeros_like(target_s)

        for _ in range(max_iter):
            r_g_iterate = wiener_khinchin_transform(s_g_iterate, self.frequency, self.time)
            for i in range(len(target_r)):
                r_ng_iterate[i] = correlation_distortion(dist_object=self.dist_object,
                                                         rho=r_g_iterate[i] / r_g_iterate[0])
            s_ng_iterate = inverse_wiener_khinchin_transform(r_ng_iterate, self.frequency, self.time)

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


def wiener_khinchin_transform(power_spectrum, frequency, time):
    """
    A function to transform the power spectrum to a correlation function by the Wiener Khinchin transformation

    ** Input:**

    * **power_spectrum** (`list or numpy.array`):

        The power spectrum of the signal.

    * **frequency** (`list or numpy.array`):

        The frequency discretizations of the power spectrum.

    * **time** (`list or numpy.array`):

        The time discretizations of the signal.

    **Output/Returns:**

    * **correlation_function** (`list or numpy.array`):

        The correlation function of the signal.

    """
    frequency_interval = frequency[1] - frequency[0]
    fac = np.ones(len(frequency))
    fac[1: len(frequency) - 1: 2] = 4
    fac[2: len(frequency) - 2: 2] = 2
    fac = fac * frequency_interval / 3
    correlation_function = np.zeros(len(time))
    for i in range(len(time)):
        correlation_function[i] = 2 * np.dot(fac, power_spectrum * np.cos(frequency * time[i]))
    return correlation_function


def inverse_wiener_khinchin_transform(correlation_function, frequency, time):
    """
    A function to transform the autocorrelation function to a power spectrum by the Inverse Wiener Khinchin
    transformation.

    ** Input:**

    * **correlation_function** (`list or numpy.array`):

        The correlation function of the signal.

    * **frequency** (`list or numpy.array`):

        The frequency discretizations of the power spectrum.

    * **time** (`list or numpy.array`):

        The time discretizations of the signal.

    **Output/Returns:**

    * **power_spectrum** (`list or numpy.array`):

        The power spectrum of the signal.
    """
    time_length = time[1] - time[0]
    fac = np.ones(len(time))
    fac[1: len(time) - 1: 2] = 4
    fac[2: len(time) - 2: 2] = 2
    fac = fac * time_length / 3
    power_spectrum = np.zeros(len(frequency))
    for i in range(len(frequency)):
        power_spectrum[i] = 2 / (2 * np.pi) * np.dot(fac, correlation_function * np.cos(time * frequency[i]))
    power_spectrum[power_spectrum < 0] = 0
    return power_spectrum


def scaling_correlation_function(correlation_function):
    """
    A function to scale a correlation function such that correlation at 0 lag is equal to 1

    ** Input:**

    * **correlation_function** (`list or numpy.array`):

        The correlation function of the signal.

    **Output/Returns:**

    * **scaled_correlation_function** (`list or numpy.array`):

        The scaled correlation functions of the signal.
    """
    scaled_correlation_function = correlation_function / np.max(correlation_function)
    return scaled_correlation_function

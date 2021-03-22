from UQpy.Utilities import *


########################################################################################################################
########################################################################################################################
#                                        Spectral Representation Method
########################################################################################################################

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

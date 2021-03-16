import itertools

from UQpy.Utilities import *


########################################################################################################################
########################################################################################################################
#                                        Bi-spectral Representation Method
########################################################################################################################

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
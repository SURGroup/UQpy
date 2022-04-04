import itertools
from UQpy.utilities import *


class BispectralRepresentation:
    def __init__(
            self,
            n_samples: int,
            power_spectrum: Union[list, np.ndarray],
            bispectrum: Union[list, np.ndarray],
            time_interval: Union[list, np.ndarray],
            frequency_interval: Union[list, np.ndarray],
            n_time_intervals: Union[list, np.ndarray],
            n_frequency_intervals: Union[list, np.ndarray],
            case="uni",
            random_state: RandomStateType = None,
    ):
        """
        A class to simulate non-Gaussian stochastic processes from a given power spectrum and bispectrum based on the
        3-rd order Spectral Representation Method. This class can simulate uni-variate, one-dimensional and
        multi-dimensional stochastic processes.

        :param n_samples: Number of samples of the stochastic process to be simulated.
         The :meth:`run` method is automatically called if `n_samples` is provided. If `n_samples` is not
         provided, then the :class:`.BispectralRepresentation` object is created but samples are not generated.
        :param power_spectrum: The discretized power spectrum.
         - For uni-variate, one-dimensional processes `power_spectrum` will be :class:`list` or :class:`numpy.ndarray`
         of length `n_frequency_intervals`.

         - For uni-variate, multi-dimensional processes, `power_spectrum` will be a :class:`list` or :class:`numpy.ndarray` of size :code:`(n_frequency_intervals[0], ..., n_frequency_intervals[n_dimensions-1])`

        :param bispectrum: The prescribed bispectrum.
         - For uni-variate, one-dimensional processes, `bispectrum` will be a :class:`list` or :class:`numpy.ndarray` of size :code:`(n_frequency_intervals, n_frequency_intervals)`

         - For uni-variate, multi-dimensional processes, `bispectrum` will be a :class:`list` or :class:`numpy.ndarray` of size :code:`(n_frequency_intervals[0], ..., n_frequency_intervals[n_dimensions-1], n_frequency_intervals[0], ..., n_frequency_intervals[n_dimensions-1])`

        :param time_interval: Length of time discretizations (:math:`\Delta t`) for each dimension of size
         `n_dimensions`.
        :param frequency_interval: Length of frequency discretizations (:math:`\Delta \omega`) for each dimension of
         size `n_dimensions`.
        :param n_time_intervals: Number of time discretizations for each dimensions of size `n_dimensions`.
        :param n_frequency_intervals: Number of frequency discretizations for each dimension of size
         `n_dimensions`.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is :any:`None`.
         If an :any:`int` is provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise,
         the object itself can be passed directly.
        """
        self.n_samples = n_samples
        self.n_frequency_intervals = np.array(n_frequency_intervals)
        self.n_time_intervals = np.array(n_time_intervals)
        self.frequency_interval = np.array(frequency_interval)
        self.time_interval = np.array(time_interval)
        self.n_dimensions: int = len(power_spectrum.shape)
        """The dimensionality of the stochastic process."""
        self.power_spectrum = power_spectrum
        self.bispectrum = bispectrum

        # Error checks
        t_u = (2 * np.pi / (2 * self.n_frequency_intervals * self.frequency_interval))
        if (self.time_interval > t_u).any():
            raise RuntimeError("UQpy: Aliasing might occur during execution")

        self.random_state = random_state
        if isinstance(self.random_state, int):
            np.random.seed(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError("UQpy: random_state must be None, an int or an np.random.RandomState object.")

        self.logger = logging.getLogger(__name__)

        self.bispectrum_amplitude: float = np.absolute(bispectrum)
        """The amplitude of the bispectrum."""
        self.bispectrum_real: float = np.real(bispectrum)
        """The real part of the bispectrum."""
        self.bispectrum_imaginary: float = np.imag(bispectrum)
        """The imaginary part of the bispectrum."""
        self.biphase: NumpyFloatArray = np.arctan2(self.bispectrum_imaginary, self.bispectrum_real)
        """The biphase values of the bispectrum."""
        self.biphase[np.isnan(self.biphase)] = 0

        self.phi: NumpyFloatArray = None
        """The random phase angles used in the simulation of the stochastic process.
        The shape of the phase angles :code:`(n_samples, n_variables, n_frequency_intervals[0], ...,
        n_frequency_intervals[n_dimensions-1])`"""
        self.samples: NumpyFloatArray = None
        """Generated samples.
        The shape of the samples is :code:`(n_samples, n_variables, n_time_intervals[0], ...,
        n_time_intervals[n_dimensions-1])`"""

        self.case = case

        if self.n_dimensions == len(self.power_spectrum.shape):
            self.case = "uni"
            self._compute_bicoherence_uni()
        else:
            self.n_variables: int = self.power_spectrum.shape[0]
            """Number of variables in the stochastic process."""
            self.case = "multi"

        if self.n_samples is not None:
            self.run(n_samples=self.n_samples)

    def _compute_bicoherence_uni(self):
        self.logger.info("UQpy: Stochastic Process: Computing the partial bicoherence values.")
        self.bc2 = np.zeros_like(self.bispectrum_real)
        """The bicoherence values of the power spectrum and bispectrum."""
        self.pure_power_sepctrum = np.zeros_like(self.power_spectrum)
        """The pure part of the power spectrum."""
        self.sum_bc2 = np.zeros_like(self.power_spectrum)
        """The sum of the bicoherence values for single frequencies."""

        if self.n_dimensions == 1:
            self.pure_power_sepctrum[0] = self.power_spectrum[0]
            self.pure_power_sepctrum[1] = self.power_spectrum[1]

        if self.n_dimensions == 2:
            self.pure_power_sepctrum[0, :] = self.power_spectrum[0, :]
            self.pure_power_sepctrum[1, :] = self.power_spectrum[1, :]
            self.pure_power_sepctrum[:, 0] = self.power_spectrum[:, 0]
            self.pure_power_sepctrum[:, 1] = self.power_spectrum[:, 1]

        if self.n_dimensions == 3:
            self.pure_power_sepctrum[0, :, :] = self.power_spectrum[0, :, :]
            self.pure_power_sepctrum[1, :, :] = self.power_spectrum[1, :, :]
            self.pure_power_sepctrum[:, 0, :] = self.power_spectrum[:, 0, :]
            self.pure_power_sepctrum[:, 1, :] = self.power_spectrum[:, 1, :]
            self.pure_power_sepctrum[:, :, 0] = self.power_spectrum[:, :, 0]
            self.pure_power_sepctrum[:, 0, 1] = self.power_spectrum[:, :, 1]

        self.ranges = [range(self.n_frequency_intervals[i]) for i in range(self.n_dimensions)]

        for i in itertools.product(*self.ranges):
            wk = np.array(i)
            for j in itertools.product(*[range(np.int32(k)) for k in np.ceil((wk + 1) / 2)]):
                wj = np.array(j)
                wi = wk - wj
                if (self.bispectrum_amplitude[(*wi, *wj)] > 0 and self.pure_power_sepctrum[(*wi, *[])]
                        * self.pure_power_sepctrum[(*wj, *[])] != 0):
                    self.bc2[(*wi, *wj)] = (self.bispectrum_amplitude[(*wi, *wj)] ** 2
                                            / (self.pure_power_sepctrum[(*wi, *[])]
                                               * self.pure_power_sepctrum[(*wj, *[])]
                                               * self.power_spectrum[(*wk, *[])]) * np.prod(self.frequency_interval))
                    self.sum_bc2[(*wk, *[])] = (self.sum_bc2[(*wk, *[])] + self.bc2[(*wi, *wj)])
                else:
                    self.bc2[(*wi, *wj)] = 0
            if self.sum_bc2[(*wk, *[])] > 1:
                self.logger.info("UQpy: Stochastic Process: Results may not be as expected as sum of partial "
                                 "bicoherences is greater than 1")
                for j in itertools.product(*[range(k) for k in np.ceil((wk + 1) / 2, dtype=np.int32)]):
                    wj = np.array(j)
                    wi = wk - wj
                    self.bc2[(*wi, *wj)] = (self.bc2[(*wi, *wj)] / self.sum_bc2[(*wk, *[])])
                self.sum_bc2[(*wk, *[])] = 1
            self.pure_power_sepctrum[(*wk, *[])] = self.power_spectrum[(*wk, *[])] * (1 - self.sum_bc2[(*wk, *[])])

    def _simulate_bsrm_uni(self, phi):
        coeff = np.sqrt((2 ** (self.n_dimensions + 1)) * self.power_spectrum * np.prod(self.frequency_interval))
        phi_e = np.exp(phi * 1.0j)
        biphase_e = np.exp(self.biphase * 1.0j)
        b = np.sqrt(1 - self.sum_bc2) * phi_e
        bc = np.sqrt(self.bc2)

        phi_e = np.einsum("i...->...i", phi_e)
        b = np.einsum("i...->...i", b)

        for i in itertools.product(*self.ranges):
            wk = np.array(i)
            for j in itertools.product(*[range(np.int32(k)) for k in np.ceil((wk + 1) / 2)]):
                wj = np.array(j)
                wi = wk - wj
                b[(*wk, *[])] = (
                        b[(*wk, *[])]
                        + bc[(*wi, *wj)]
                        * biphase_e[(*wi, *wj)]
                        * phi_e[(*wi, *[])]
                        * phi_e[(*wj, *[])])

        b = np.einsum("...i->i...", b)
        b = b * coeff
        b[np.isnan(b)] = 0
        samples = np.fft.fftn(b, self.n_time_intervals)
        samples = samples[:, np.newaxis]
        return np.real(samples)

    def run(self, n_samples: int):
        """
        Execute the random sampling in the :class:`.BispectralRepresentation` class.

        The :meth:`run` method is the function that performs random sampling in the :class:`.BispectralRepresentation`
        class. If `n_samples` is provided, the :meth:`run` method is automatically called when the
        :class:`.BispectralRepresentation` object is defined. The user may also call the :meth:`run` method directly to
        generate samples. The :meth:`run` method of the :class:`.BispectralRepresentation` class can be invoked many
        times and each time the generated samples are appended to the existing samples.

        :param n_samples: Number of samples of the stochastic process to be simulated.
         If the :meth:`run` method is invoked multiple times, the newly generated samples will be appended to the
         existing samples.

        The :meth:`run` method has no returns, although it creates and/or appends the :py:attr:`samples` attribute of
        the :class:`.BispectralRepresentation` class.
        """
        if n_samples is None:
            raise ValueError("UQpy: Stochastic Process: Number of samples must be defined.")
        if not isinstance(n_samples, int):
            raise ValueError("UQpy: Stochastic Process: n_samples should be an integer.")

        self.logger.info("UQpy: Stochastic Process: Running 3rd-order Spectral Representation Method.")

        samples = None
        phi = None

        if self.case == "uni":
            self.logger.info("UQpy: Stochastic Process: Starting simulation of uni-variate Stochastic Processes.")
            self.logger.info("UQpy: The number of dimensions is %i:", self.n_dimensions)
            phi = (np.random.uniform(
                size=np.append(self.n_samples,
                               np.ones(self.n_dimensions, dtype=np.int32)
                               * self.n_frequency_intervals, )) * 2 * np.pi)
            samples = self._simulate_bsrm_uni(phi)

        if self.samples is None:
            self.samples = samples
            self.phi = phi
        else:
            self.samples = np.concatenate((self.samples, samples), axis=0)
            self.phi = np.concatenate((self.phi, phi), axis=0)

        self.logger.info("UQpy: Stochastic Process: 3rd-order Spectral Representation Method Complete.")

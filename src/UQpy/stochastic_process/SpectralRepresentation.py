from UQpy.utilities import *
import numpy as np


class SpectralRepresentation:
    def __init__(
            self,
            n_samples: int,
            power_spectrum: Union[list, np.ndarray, float],
            time_interval: Union[list, np.ndarray, float],
            frequency_interval: Union[list, np.ndarray, float],
            n_time_intervals: Union[list, np.ndarray, float],
            n_frequency_intervals: Union[list, np.ndarray, float],
            random_state: RandomStateType = None,
    ):
        """
        A class to simulate stochastic processes from a given power spectrum density using the Spectral Representation
        Method. This class can simulate uni-variate, multi-variate, and multi-dimensional stochastic processes. The
        class uses Singular Value Decomposition, as opposed to Cholesky Decomposition, to ensure robust, near-positive
        definite multi-dimensional power spectra.

        :param n_samples: Number of samples of the stochastic process to be simulated. The :py:meth:`run` method
         is automatically called if `n_samples` is provided. If `n_samples` is not provided, then the
         :class:`.SpectralRepresentation` object is created but samples are not generated.
        :param power_spectrum: The discretized power spectrum.

         * For uni-variate, one-dimensional processes `power_spectrum` will be :class:`list` or :class:`numpy.ndarray` of length `n_frequency_intervals`.

         * For multi-variate, one-dimensional processes, `power_spectrum` will be a :class:`list` or :class:`numpy.ndarray` of size :code:`(n_of_variables, n_variables, n_frequency_intervals)`.

         * For uni-variate, multi-dimensional processes, `power_spectrum` will be a :class:`list` or :class:`numpy.ndarray` of size :code:`(n_frequency_intervals[0], ..., n_frequency_intervals[n_dimensions-1])`

         * For multi-variate, multi-dimensional processes, `power_spectrum` will be a :class:`list` or :class:`numpy.ndarray` of size :code:`(n_variables, n_variables, n_frequency_intervals[0],...,n_frequency_intervals[n_dimensions-1])`.

        :param time_interval: Length of time discretizations (:math:`\Delta t`) for each
         dimension of size `n_dimensions`.
        :param frequency_interval: Length of frequency discretizations
         (:math:`\Delta \omega`) for each dimension of size `n_dimensions`.
        :param n_time_intervals: Number of time discretizations for each dimensions of
         size `n_dimensions`.
        :param n_frequency_intervals: Number of frequency discretizations for each
         dimension of size `n_dimensions`.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is :any:`None`.
         If an :any:`int` is provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise,
         the object itself can be passed directly.
        """
        self.power_spectrum = power_spectrum
        if (
                isinstance(time_interval, float)
                and isinstance(frequency_interval, float)
                and isinstance(n_time_intervals, int)
                and isinstance(n_frequency_intervals, int)
        ):
            time_interval = [time_interval]
            frequency_interval = [frequency_interval]
            n_time_intervals = [n_time_intervals]
            n_frequency_intervals = [n_frequency_intervals]
        self.time_interval = np.array(time_interval)
        self.frequency_interval = np.array(frequency_interval)
        self.number_time_intervals = np.array(n_time_intervals)
        self.number_frequency_intervals = np.array(n_frequency_intervals)
        self.n_samples = n_samples

        # Error checks
        t_u = 2 * np.pi / (2 * self.number_frequency_intervals * self.frequency_interval)

        if (self.time_interval > t_u).any():
            raise RuntimeError("UQpy: Aliasing might occur during execution")

        self.logger = logging.getLogger(__name__)

        self.random_state = random_state
        if isinstance(self.random_state, int):
            np.random.seed(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError("UQpy: random_state must be None, an int or an np.random.RandomState object.")

        self.samples: NumpyFloatArray = None
        """Generated samples.
        The shape of the samples is :code:`(n_samples, n_variables, n_time_intervals[0], ...,
        n_time_intervals[n_dimensions-1])`"""
        self.n_variables: int = None
        """Number of variables in the stochastic process."""
        self.n_dimensions: int = len(self.number_frequency_intervals)
        """The dimensionality of the stochastic process."""
        self.phi: NumpyFloatArray = None
        """The random phase angles used in the simulation of the stochastic process.
        The shape of the phase angles :code:`(n_samples, n_variables, n_frequency_intervals[0], ...,
        n_frequency_intervals[n_dimensions-1])`"""

        if self.n_dimensions == len(self.power_spectrum.shape):
            self.case = "uni"
        else:
            self.n_variables = self.power_spectrum.shape[0]
            self.case = "multi"

        # Run Spectral Representation Method
        if self.n_samples is not None:
            self.run(n_samples=self.n_samples)

    def run(self, n_samples):
        """
        Execute the random sampling in the :class:`.SpectralRepresentation` class.

        The :meth:`run` method is the function that performs random sampling in the :class:`.SpectralRepresentation`
        class. If `n_samples` is provided when the :class:`.SpectralRepresentation` object is defined, the
        :meth:`run` method is automatically called. The user may also call the :meth:`run` method directly to generate
        samples. The :meth:`run` method of the :class:`.SpectralRepresentation` class can be invoked many times and each
        time the generated samples are appended to the existing samples.

        :param int n_samples: Number of samples of the stochastic process to be simulated.
         If the :meth:`run` method is invoked multiple times, the newly generated samples will be appended to the
         existing samples.

        The :meth:`run` method has no returns, although it creates and/or appends the :py:attr:`samples` attribute of
        the :class:`.SpectralRepresentation` class.
        """
        if n_samples is None:
            raise ValueError("UQpy: Stochastic Process: Number of samples must be defined.")
        if not isinstance(n_samples, int):
            raise ValueError("UQpy: Stochastic Process: nsamples should be an integer.")

        self.logger.info("UQpy: Stochastic Process: Running Spectral Representation Method.")

        samples = None
        phi = None

        if self.case == "uni":
            self.logger.info("UQpy: Stochastic Process: Starting simulation of uni-variate Stochastic Processes.")
            self.logger.info("UQpy: The number of dimensions is %i:", self.n_dimensions)
            phi = (np.random.uniform(size=np.append(self.n_samples, np.ones(self.n_dimensions, dtype=np.int32)
                                                    * self.number_frequency_intervals, )) * 2 * np.pi)
            samples = self._simulate_uni(phi)

        elif self.case == "multi":
            self.logger.info("UQpy: Stochastic Process: Starting simulation of multi-variate Stochastic Processes.")
            self.logger.info("UQpy: Stochastic Process: The number of variables is %i:", self.n_variables)
            self.logger.info("UQpy: Stochastic Process: The number of dimensions is  %i:", self.n_dimensions)
            phi = (np.random.uniform(size=
                                     np.append(self.n_samples, np.append(np.ones(self.n_dimensions,
                                                                                 dtype=np.int32)
                                                                         * self.number_frequency_intervals,
                                                                         self.n_variables, ), )) * 2 * np.pi)
            samples = self._simulate_multi(phi)

        if self.samples is None:
            self.samples = samples
            self.phi = phi
        else:
            self.samples = np.concatenate((self.samples, samples), axis=0)
            self.phi = np.concatenate((self.phi, phi), axis=0)

        self.logger.info("UQpy: Stochastic Process: Spectral Representation Method Complete.")

    def _simulate_uni(self, phi):
        fourier_coefficient = np.exp(phi * 1.0j) * np.sqrt(
            2 ** (self.n_dimensions + 1) * self.power_spectrum * np.prod(self.frequency_interval))
        samples = np.fft.fftn(fourier_coefficient, s=self.number_time_intervals)
        samples = np.real(samples)
        samples = samples[:, np.newaxis]
        return samples

    def _simulate_multi(self, phi):
        power_spectrum = np.einsum("ij...->...ij", self.power_spectrum)
        coefficient = np.sqrt(2 ** (self.n_dimensions + 1)) * np.sqrt(np.prod(self.frequency_interval))
        u, s, v = np.linalg.svd(power_spectrum)
        power_spectrum_decomposed = np.einsum("...ij,...j->...ij", u, np.sqrt(s))
        fourier_coefficient = coefficient * np.einsum(
            "...ij,n...j -> n...i", power_spectrum_decomposed, np.exp(phi * 1.0j))
        fourier_coefficient[np.isnan(fourier_coefficient)] = 0
        samples = np.real(np.fft.fftn(fourier_coefficient, s=self.number_time_intervals,
                                      axes=tuple(np.arange(1, 1 + self.n_dimensions))))
        samples = np.einsum("n...m->nm...", samples)
        return samples

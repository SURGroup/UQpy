import logging
import numpy as np
from beartype import beartype
from typing import Optional
from UQpy.utilities.ValidationTypes import (
    NumericArrayLike,
    NumpyFloatArray,
    PositiveInteger,
    RandomStateType,
)


@beartype
class SpectralRepresentation:
    def __init__(
        self,
        power_spectrum: NumericArrayLike,
        time_interval: NumericArrayLike,
        frequency_interval: NumericArrayLike,
        n_time_intervals: NumericArrayLike,
        n_frequency_intervals: NumericArrayLike,
        n_samples: Optional[PositiveInteger] = None,
        phase_angles: Optional[NumericArrayLike] = None,
        random_state: Optional[RandomStateType] = None,
    ):
        """A class to simulate stochastic processes from a given power spectrum density using the
        Spectral Representation Method :cite:`StochasticProcess2`.

        This class can simulate uni-variate, multi-variate, and multi-dimensional stochastic processes.
        The class uses Singular Value Decomposition, as opposed to Cholesky Decomposition, to ensure robust,
        near-positive definite multi-dimensional power spectra.
        This class checks if the criteria :math:`\Delta t \leq 2\pi / 2\omega_u` is met, and raises a
        :code:`RuntimeError` if the inequality is violated.

        :param power_spectrum: The discretized power spectrum.

         * For uni-variate, one-dimensional processes, ``power_spectrum`` has length ``n_frequency_intervals``.

         * For multi-variate, one-dimensional processes, ``power_spectrum`` has size
         :code:`(n_variables, n_variables, n_frequency_intervals)`.

         * For uni-variate, multi-dimensional processes, ``power_spectrum`` has size
         :code:`(n_frequency_intervals[0], ..., n_frequency_intervals[n_dimensions-1])`

         * For multi-variate, multi-dimensional processes, ``power_spectrum`` has size
         :code:`(n_variables, n_variables, n_frequency_intervals[0],...,n_frequency_intervals[n_dimensions-1])`.

        :param time_interval: Length of time discretizations
         (:math:`\Delta t`) for each dimension of size :code:`n_dimensions`.
        :param frequency_interval: Length of frequency discretizations
         (:math:`\Delta \omega`) for each dimension of size :code:`n_dimensions`.
        :param n_time_intervals: Number of time discretizations
         for each dimensions of size :code:`n_dimensions`.
        :param n_frequency_intervals: Number of frequency discretizations
         for each dimension of size :code:`n_dimensions`.
        :param n_samples: Number of samples of the stochastic process to be simulated.
         The :py:meth:`run` method is automatically called if :code:`n_samples` is provided.
         If :code:`n_samples` is not provided, then the :class:`.SpectralRepresentation` object is created
         but samples are not generated.
        :param phase_angles: Optional, phase angles (:math:`\Phi`) used in the Spectral Representation Method.
         If :code:`phase_angles` is not provided,
         they are randomly generated as i.i.d. uniform random variables :math:`\phi_i\sim Uniform(0, 2\pi)`.
        The shape of the phase angles is
         :code:`(n_samples, n_variables, n_frequency_intervals[0], ..., n_frequency_intervals[n_dimensions-1])
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is :code:`None`.
         If an :code:`int` or :code:`np.random.RandomState` is provided, this sets :py:meth:`np.random.seed`.
        """
        self.power_spectrum = np.atleast_1d(power_spectrum)
        self.time_interval = np.atleast_1d(time_interval)
        self.frequency_interval = np.atleast_1d(frequency_interval)
        self.n_time_intervals = np.atleast_1d(n_time_intervals)
        self.n_frequency_intervals = np.atleast_1d(n_frequency_intervals)
        self.n_samples = n_samples
        self.phase_angles = phase_angles
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.n_dimensions: int = len(self.n_frequency_intervals)
        """The dimensionality of the stochastic process."""
        self.n_variables: int = (
            1
            if self.n_dimensions == self.power_spectrum.ndim
            else self.power_spectrum.shape[0]
        )
        """Number of variables in the stochastic process."""
        self.samples: NumpyFloatArray = None
        """Generated samples. The shape of the samples is 
         :code:`(n_samples, n_variables, n_time_intervals[0], ..., n_time_intervals[n_dimensions-1])`"""
        self.logger = logging.getLogger(__name__)

        # Check if Equation 45 from Shinozuka and Deodatis 1991 is satisfied
        frequency_cutoff = self.frequency_interval * self.n_frequency_intervals
        max_time_interval = 2 * np.pi / (2 * frequency_cutoff)
        if (self.time_interval > max_time_interval).any():
            raise RuntimeError(
                "UQpy: time_interval greater than pi / cutoff_frequency. Aliasing might occur during execution."
            )

        if (self.n_samples is None) and (self.phase_angles is not None):
            raise RuntimeError(
                "UQpy: Specifying `phase_angles` without `n_samples` may cause unintended results."
                "Either specify both on initialization or both via the `run` method."
            )

        if self.n_samples is not None:
            self.run(n_samples=self.n_samples, phase_angles=self.phase_angles)

    def run(
        self, n_samples: PositiveInteger, phase_angles: Optional[np.ndarray] = None
    ):
        """Execute the random sampling in the :class:`.SpectralRepresentation` class.

         If ``n_samples`` is upon :class:`.SpectralRepresentation` initialization, :meth:`run` is automatically called.
         Alternatively, call the :meth:`run` method directly to generate samples.
         The :meth:`run` method may be called many times and the generated samples are appended to the existing samples.

        :param n_samples: Number of samples of the stochastic process to be simulated.
        :param phase_angles: Optional, phase angles (:math:`\Phi`) used in the Spectral Representation Method.
         If :code:`phase_angles` is not provided, they are randomly generated as i.i.d. uniform random variables :math:`\phi_i\sim Uniform(0, 2\pi)`.

        The :meth:`run` method has no returns, although it creates and/or appends the :py:attr:`samples` attribute of
        the :class:`.SpectralRepresentation` class.
        """
        self.n_samples = n_samples
        if phase_angles is None:
            phase_angles = np.random.uniform(
                low=0,
                high=2 * np.pi,
                size=[
                    self.n_samples,
                    self.n_variables,
                    *self.n_frequency_intervals,
                ],
            )

        if self.n_variables == 1:  # uni-variate case
            self.logger.info(
                "UQpy: Stochastic Process: Starting simulation of uni-variate Spectral Representation Method."
            )
            self.logger.info(
                f"UQpy: Stochastic Process: The number of dimensions is {self.n_dimensions}",
            )
            samples = self._simulate_univariate(phase_angles)
        else:  # multi-variate case
            self.logger.info(
                "UQpy: Stochastic Process: Starting simulation of multi-variate Spectral Representation Method."
            )
            self.logger.info(
                f"UQpy: Stochastic Process: The number of variables is {self.n_variables}:"
            )
            self.logger.info(
                f"UQpy: Stochastic Process: The number of dimensions is {self.n_dimensions}:"
            )
            # if phase_angles is None:
            #     size = np.append(
            #         self.n_samples,
            #         np.append(self.n_frequency_intervals, self.n_variables),
            #     )
            #     phase_angles = np.random.uniform(low=0, high=2 * np.pi, size=size)
            samples = self._simulate_multivariate(phase_angles)

        if self.samples is None:
            self.samples = samples
        else:
            self.samples = np.concatenate((self.samples, samples), axis=0)

        if self.phase_angles is None:
            self.phase_angles = phase_angles
        else:
            self.phase_angles = np.concatenate(
                (self.phase_angles, phase_angles), axis=0
            )

        self.logger.info(
            "UQpy: Stochastic Process: Spectral Representation Method Complete."
        )

    def _simulate_univariate(self, phase_angles: np.ndarray) -> np.ndarray:
        """Simulate univariate random processes using spectral representation method

        :param phase_angles: Phase angles :math:`\Phi` used in the spectral representation method
        :return: Random samples computed using spectral representation method
        """
        # fourier_coefficient = (
        #     2
        #     * np.exp(phase_angles * 1.0j)
        #     * np.sqrt(
        #         # 2 ** (self.n_dimensions + 1)  # ToDo: Is there any reason this should scale with dimension?
        #         self.power_spectrum
        #         * np.prod(self.frequency_interval)
        #     )
        # )
        fourier_coefficient = np.sqrt(self.power_spectrum) * np.exp(phase_angles * 1.0j)
        samples = np.real(np.fft.ifftn(fourier_coefficient, s=self.n_time_intervals))
        time_domain_coefficient = (
            2
            * np.prod(self.n_time_intervals)
            * np.sqrt(np.prod(self.frequency_interval))
        )
        return time_domain_coefficient * samples

    def _simulate_multivariate(self, phase_angles: np.ndarray) -> np.ndarray:
        """Simulate multivariate random process using spectral representation method

        :param phase_angles: Phase angles :math:`\Phi` used in the spectral representation method
        :return: Random samples computed using spectral representation method
        """
        power_spectrum = np.einsum("ij...->...ij", self.power_spectrum)
        coefficient = np.sqrt(2 ** (self.n_dimensions + 1)) * np.sqrt(
            np.prod(self.frequency_interval)
        )
        u, s, v = np.linalg.svd(power_spectrum)
        power_spectrum_decomposed = np.einsum("...ij,...j->...ij", u, np.sqrt(s))
        fourier_coefficient = coefficient * np.einsum(
            "...ij,n...j -> n...i",
            power_spectrum_decomposed,
            np.exp(phase_angles * 1.0j),
        )
        fourier_coefficient[np.isnan(fourier_coefficient)] = 0
        samples = np.real(
            np.fft.fftn(
                fourier_coefficient,
                s=self.n_time_intervals,
                axes=tuple(np.arange(1, 1 + self.n_dimensions)),
            )
        )
        samples = np.einsum("n...m->nm...", samples)
        return samples

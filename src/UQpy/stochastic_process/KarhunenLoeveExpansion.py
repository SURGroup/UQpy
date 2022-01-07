from scipy.linalg import sqrtm
import numpy as np
from UQpy.utilities import *


class KarhunenLoeveExpansion:
    # TODO: Test this for non-stationary processes.

    def __init__(
        self,
        samples_number: int,
        correlation_function: np.ndarray,
        time_interval: np.ndarray,
        threshold: int = None,
        random_state: RandomStateType = None,
    ):
        """
        A class to simulate stochastic processes from a given auto-correlation function based on the Karhunen-Loeve
        Expansion

        :param samples_number: Number of samples of the stochastic process to be simulated.
         The :meth:`run` method is automatically called if `samples_number` is provided. If `samples_number` is not
         provided, then the :class:`.KarhunenLoeveExpansion` object is created but samples are not generated.
        :param correlation_function: The correlation function of the stochastic process of size
         :code:`(number_time_intervals, number_time_intervals)`
        :param time_interval: The length of time discretization.
        :param threshold: The threshold number of eigenvalues to be used in the expansion.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is :any:`None`.

        """
        self.correlation_function = correlation_function
        self.time_interval = time_interval
        if threshold:
            self.number_eigenvalues = threshold
        else:
            self.number_eigenvalues = len(self.correlation_function[0])

        self.random_state = random_state
        if isinstance(self.random_state, int):
            np.random.seed(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError(
                "UQpy: random_state must be None, an int or an np.random.RandomState object."
            )

        self.logger = logging.getLogger(__name__)
        self.nsamples = samples_number

        self.samples: NumpyFloatArray = None
        """Array of generated samples."""
        self.xi: NumpyFloatArray = None
        """The independent gaussian random variables used in the expansion."""

        if self.nsamples is not None:
            self.run(samples_number=self.nsamples)

    def _simulate(self, xi):
        lam, phi = np.linalg.eig(self.correlation_function)
        lam = lam[: self.number_eigenvalues]
        lam = np.diag(lam)
        self.phi = np.real(phi[:, : self.number_eigenvalues])
        self.lam = lam.astype(np.float64)
        samples = np.dot(self.phi, np.dot(sqrtm(self.lam), xi))
        samples = np.real(samples)
        samples = samples.T
        samples = samples[:, np.newaxis]
        return samples

    def run(self, samples_number):
        """
        Execute the random sampling in the :class:`.KarhunenLoeveExpansion` class.

        The :meth:`run` method is the function that performs random sampling in the :class:`.KarhunenLoeveExpansion``
        class. If `samples_number` is provided when the :class:`.KarhunenLoeveExpansion` object is defined, the
        :meth:`run` method is automatically called. The user may also call the :meth:`run` method directly to generate
        samples. The :meth:`run`` method of the :class:`.KarhunenLoeveExpansion` class can be invoked many times and
        each time the generated samples are appended to the existing samples.

        :param samples_number: Number of samples of the stochastic process to be simulated.
         If the :meth:`run` method is invoked multiple times, the newly generated samples will be appended to the
         existing samples.

        The :meth:`run` method has no returns, although it creates and/or appends the `samples` attribute of the
        :class:`KarhunenLoeveExpansion` class.
        """
        if samples_number is None:
            raise ValueError(
                "UQpy: Stochastic Process: Number of samples must be defined."
            )
        if not isinstance(samples_number, int):
            raise ValueError("UQpy: Stochastic Process: nsamples should be an integer.")

        self.logger.info("UQpy: Stochastic Process: Running Karhunen Loeve Expansion.")

        self.logger.info(
            "UQpy: Stochastic Process: Starting simulation of Stochastic Processes."
        )
        xi = np.random.normal(size=(self.number_eigenvalues, self.nsamples))
        samples = self._simulate(xi)

        if self.samples is None:
            self.samples = samples
            self.xi = xi
        else:
            self.samples = np.concatenate((self.samples, samples), axis=0)
            self.xi = np.concatenate((self.xi, xi), axis=0)

        self.logger.info("UQpy: Stochastic Process: Karhunen-Loeve Expansion Complete.")

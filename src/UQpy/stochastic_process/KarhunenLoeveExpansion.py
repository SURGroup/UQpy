from scipy.linalg import sqrtm
import numpy as np
from UQpy.utilities import *


class KarhunenLoeveExpansion:
    # TODO: Test this for non-stationary processes.

    def __init__(
        self,
        n_samples: int,
        correlation_function: np.ndarray,
        time_interval: Union[np.ndarray, float],
        threshold: int = None,
        random_state: RandomStateType = None,
    ):
        """
        A class to simulate stochastic processes from a given auto-correlation function based on the Karhunen-Loeve
        Expansion

        :param n_samples: Number of samples of the stochastic process to be simulated.
         The :meth:`run` method is automatically called if `n_samples` is provided. If `n_samples` is not
         provided, then the :class:`.KarhunenLoeveExpansion` object is created but samples are not generated.
        :param correlation_function: The correlation function of the stochastic process of size
         :code:`(n_time_intervals, n_time_intervals)`
        :param time_interval: The length of time discretization.
        :param threshold: The threshold number of eigenvalues to be used in the expansion.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is :any:`None`.

        """
        self.correlation_function = correlation_function
        self.time_interval = time_interval
        self.n_eigenvalues = threshold or len(self.correlation_function[0])
        self.random_state = random_state
        if isinstance(self.random_state, int):
            np.random.seed(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError("UQpy: random_state must be None, an int or an np.random.RandomState object.")

        self.logger = logging.getLogger(__name__)
        self.n_samples = n_samples

        self.samples: NumpyFloatArray = None
        """Array of generated samples."""
        self.xi: NumpyFloatArray = None
        """The independent gaussian random variables used in the expansion."""

        if self.n_samples is not None:
            self.run(n_samples=self.n_samples)

    def _simulate(self, xi):
        lam, phi = np.linalg.eig(self.correlation_function)
        lam = lam[: self.n_eigenvalues]
        lam = np.diag(lam)
        self.phi = np.real(phi[:, : self.n_eigenvalues])
        self.lam = lam.astype(np.float64)
        samples = np.dot(self.phi, np.dot(sqrtm(self.lam), xi))
        samples = np.real(samples)
        samples = samples.T
        samples = samples[:, np.newaxis]
        return samples

    def run(self, n_samples: int):
        """
        Execute the random sampling in the :class:`.KarhunenLoeveExpansion` class.

        The :meth:`run` method is the function that performs random sampling in the :class:`.KarhunenLoeveExpansion`
        class. If `n_samples` is provided when the :class:`.KarhunenLoeveExpansion` object is defined, the
        :meth:`run` method is automatically called. The user may also call the :meth:`run` method directly to generate
        samples. The :meth:`run`` method of the :class:`.KarhunenLoeveExpansion` class can be invoked many times and
        each time the generated samples are appended to the existing samples.

        :param n_samples: Number of samples of the stochastic process to be simulated.
         If the :meth:`run` method is invoked multiple times, the newly generated samples will be appended to the
         existing samples.

        The :meth:`run` method has no returns, although it creates and/or appends the :py:attr:`samples` attribute of
        the :class:`KarhunenLoeveExpansion` class.
        """
        if n_samples is None:
            raise ValueError("UQpy: Stochastic Process: Number of samples must be defined.")
        if not isinstance(n_samples, int):
            raise ValueError("UQpy: Stochastic Process: n_samples should be an integer.")

        self.logger.info("UQpy: Stochastic Process: Running Karhunen Loeve Expansion.")

        self.logger.info("UQpy: Stochastic Process: Starting simulation of Stochastic Processes.")
        xi = np.random.normal(size=(self.n_eigenvalues, self.n_samples))
        samples = self._simulate(xi)

        if self.samples is None:
            self.samples = samples
            self.xi = xi
        else:
            self.samples = np.concatenate((self.samples, samples), axis=0)
            self.xi = np.concatenate((self.xi, xi), axis=0)

        self.logger.info("UQpy: Stochastic Process: Karhunen-Loeve Expansion Complete.")

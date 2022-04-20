from scipy.linalg import sqrtm
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
            random_variables: np.ndarray = None,
    ):
        """
        A class to simulate stochastic processes from a given auto-correlation function based on the Karhunen-Loeve
        Expansion

        :param n_samples: Number of samples of the stochastic process to be simulated.
         The :meth:`run` method is automatically called if `n_samples` is provided. If `n_samples` is not
         provided, then the :class:`.KarhunenLoeveExpansion` object is created but samples are not generated.
        :param correlation_function: The correlation function of the stochastic process of size
         :code:`(n_time_intervals, n_time_intervals)`
        :param time_interval: The length of time interval.
        :param threshold: The threshold number of eigenvalues to be used in the expansion.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is :any:`None`.
        :param random_variables: The random variables used to generate the stochastic process. Default is :any:`None`.
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
            self.run(n_samples=self.n_samples, random_variables=random_variables)

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

    def run(self, n_samples: int, random_variables: np.ndarray = None):
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

        if random_variables is not None and random_variables.shape == (self.n_eigenvalues, self.n_samples):
            self.logger.info('UQpy: Stochastic Process: Using user defined random variables')
        else:
            self.logger.info('UQpy: Stochastic Process; Using computer generated random variables')
            random_variables = np.random.normal(size=(self.n_eigenvalues, self.n_samples))

        # xi = np.random.normal(size=(self.n_eigenvalues, self.n_samples))
        samples = self._simulate(random_variables)

        if self.samples is None:
            self.samples = samples
            self.random_variables = random_variables
        else:
            self.samples = np.concatenate((self.samples, samples), axis=0)
            self.random_variables = np.concatenate((self.random_variables, random_variables), axis=0)

        self.logger.info("UQpy: Stochastic Process: Karhunen-Loeve Expansion Complete.")


class KarhunenLoeveExpansionTwoDimension:

    def __init__(
            self,
            n_samples: int,
            correlation_function: np.ndarray,
            time_intervals: Union[np.ndarray, float],
            thresholds: Union[list, int] = None,
            random_state: RandomStateType = None,
            random_variables=None
    ):
        """
        A class to simulate two dimensional stochastic fields from a given auto-correlation function based on the
        Karhunen-Loeve Expansion

        :param n_samples: Number of samples of the stochastic field to be simulated.
         The :meth:`run` method is automatically called if `n_samples` is provided. If `n_samples` is not
         provided, then the :class:`.KarhunenLoeveExpansionTwoDimension` object is created but samples are not generated.
        :param correlation_function: The correlation function of the stochastic process of size
         :code:`(n_time_intervals_dim1, n_time_intervals_dim1, n_time_intervals_dim2, n_time_intervals_dim2)`
        :param time_intervals: The length of time discretizations.
        :param thresholds: The threshold number of eigenvalues to be used in the expansion for two dimensions.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is :any:`None`.
        :param random_variables: The random variables used to generate the stochastic field.
        """

        self.n_samples = n_samples
        self.correlation_function = correlation_function
        assert (len(self.correlation_function.shape) == 4)
        self.time_intervals = time_intervals
        self.thresholds = thresholds
        self.random_state = random_state

        if isinstance(self.random_state, int):
            np.random.seed(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')

        self.samples = None
        self.random_variables = random_variables

        self._precompute_one_dimensional_correlation_function()

        if self.n_samples is not None:
            self.run(n_samples=self.n_samples, random_variables=random_variables)

    def _precompute_one_dimensional_correlation_function(self):
        self.quasi_correlation_function = np.diagonal(self.correlation_function, axis1=0, axis2=1)
        self.quasi_correlation_function = np.einsum('ij... -> ...ij', self.quasi_correlation_function)
        self.w, self.v = np.linalg.eig(self.quasi_correlation_function)
        if np.linalg.norm(np.imag(self.w)) > 0:
            print('Complex in the eigenvalues, check the positive definiteness')
        self.w = np.real(self.w)
        self.v = np.real(self.v)
        if self.thresholds is not None:
            self.w = self.w[:, :self.thresholds[1]]
            self.v = self.v[:, :, :self.thresholds[1]]
        self.one_dimensional_correlation_function = np.einsum('uvxy, uxn, vyn, un, vn -> nuv',
                                                              self.correlation_function, self.v, self.v,
                                                              1 / np.sqrt(self.w), 1 / np.sqrt(self.w))

    def run(self, n_samples, random_variables=None):
        """
        Execute the random sampling in the :class:`.KarhunenLoeveExpansion` class.

        The :meth:`run` method is the function that performs random sampling in the :class:
        `.KarhunenLoeveExpansionTwoDimension` class. If `n_samples` is provided when the :class:
        `.KarhunenLoeveExpansionTwoDimension` object is defined, the :meth:`run` method is automatically called. The
        user may also call the :meth:`run` method directly to generate samples. The :meth:`run`` method of the :class:
        `.KarhunenLoeveExpansionTwoDimension` class can be invoked many times and each time the generated samples are
        appended to the existing samples.

        :param n_samples: Number of samples of the stochastic process to be simulated.
         If the :meth:`run` method is invoked multiple times, the newly generated samples will be appended to the
         existing samples.

        The :meth:`run` method has no returns, although it creates and/or appends the :py:attr:`samples` attribute of
        the :class:`KarhunenLoeveExpansionTwoDimension` class.
        """
        samples = np.zeros((n_samples, self.correlation_function.shape[0], self.correlation_function.shape[2]))
        if random_variables is None:
            random_variables = np.random.normal(size=[self.thresholds[1], self.thresholds[0], n_samples])
        else:
            assert (random_variables.shape == (self.thresholds[1], self.thresholds[0], n_samples))
        for i in range(self.one_dimensional_correlation_function.shape[0]):
            if self.thresholds is not None:
                print(self.w.shape)
                print(self.v.shape)
                samples += np.einsum('x, xt, nx -> nxt', np.sqrt(self.w[:, i]), self.v[:, :, i],
                                     KarhunenLoeveExpansion(n_samples=n_samples,
                                                            correlation_function=
                                                            self.one_dimensional_correlation_function[i],
                                                            time_interval=self.time_intervals,
                                                            threshold=self.thresholds[0],
                                                            random_variables=random_variables[i]).samples[:, 0, :])
            else:
                samples += np.einsum('x, xt, nx -> nxt', np.sqrt(self.w[:, i]), self.v[:, :, i],
                                     KarhunenLoeveExpansion(n_samples=n_samples,
                                                            correlation_function=
                                                            self.one_dimensional_correlation_function[i],
                                                            time_interval=self.time_intervals,
                                                            random_variables=random_variables[i]).samples[:, 0, :])
        samples = np.reshape(samples, [samples.shape[0], 1, samples.shape[1], samples.shape[2]])

        if self.samples is None:
            self.samples = samples
            self.random_variables = random_variables
        else:
            self.samples = np.concatenate((self.samples, samples), axis=0)
            self.random_variables = np.concatenate((self.random_variables, random_variables), axis=2)

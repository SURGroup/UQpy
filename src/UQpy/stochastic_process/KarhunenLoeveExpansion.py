import logging
from typing import Union

import numpy as np
from beartype import beartype
from scipy.linalg import sqrtm

from UQpy.utilities.Utilities import *
from UQpy.utilities.ValidationTypes import *


class KarhunenLoeveExpansion:
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
    @beartype
    def __init__(self,
                 samples_number: Union[PositiveInteger, int],
                 correlation_function: Union[list, np.ndarray],
                 time_interval: float,
                 threshold: int = None,
                 random_state: RandomStateType = None):
        self.correlation_function = correlation_function
        self.time_interval = time_interval
        self.number_eigen_values = threshold if threshold else len(self.correlation_function[0])
        self.random_state = process_random_state(random_state)
        self.logger = logging.getLogger(__name__)
        self.samples_number = samples_number
        self.samples = None
        self.xi = None

        if self.samples_number is not None:
            self.run(samples_number=self.samples_number)

    def _simulate(self, xi):
        lam, phi = np.linalg.eig(self.correlation_function)
        lam = lam[:self.number_eigen_values]
        lam = np.diag(lam)
        lam = lam.astype(np.float64)
        samples = np.dot(phi[:, :self.number_eigen_values], np.dot(sqrtm(lam), xi))
        samples = np.real(samples)
        samples = samples.T
        samples = samples[:, np.newaxis]
        return samples

    def run(self, samples_number: PositiveInteger):
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
        self.logger.info('UQpy: Stochastic Process: Running Karhunen Loeve Expansion.')
        self.logger.info('UQpy: Stochastic Process: Starting simulation of Stochastic Processes.')
        xi = np.random.normal(size=(self.number_eigen_values, samples_number))
        samples = self._simulate(xi)

        if self.samples is None:
            self.samples = samples
            self.xi = xi
        else:
            self.samples = np.concatenate((self.samples, samples), axis=0)
            self.xi = np.concatenate((self.xi, xi), axis=0)

        self.logger.info('UQpy: Stochastic Process: Karhunen-Loeve Expansion Complete.')

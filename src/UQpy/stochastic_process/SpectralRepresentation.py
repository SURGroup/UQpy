import logging

import numpy as np
from beartype import beartype

from UQpy.stochastic_process.supportive.MultivariateStochasticProcess import MultivariateStochasticProcess
from UQpy.stochastic_process.supportive.UnivariateStochacticProcess import UnivariateStochasticProcess
from UQpy.utilities.Utilities import *
from UQpy.utilities.ValidationTypes import *


class SpectralRepresentationMethod:
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
    @beartype
    def __init__(self,
                 samples_number: PositiveInteger,
                 power_spectrum: Union[list, np.ndarray],
                 time_interval: Union[list, np.ndarray],
                 frequency_interval: Union[list, np.ndarray],
                 time_intervals_number: int,
                 frequency_intervals_number: int,
                 random_state: RandomStateType = None):
        self.power_spectrum = power_spectrum
        if isinstance(time_interval, float) and isinstance(frequency_interval, float) and \
                isinstance(time_intervals_number, int) and isinstance(frequency_intervals_number, int):
            time_interval = [time_interval]
            frequency_interval = [frequency_interval]
            time_intervals_number = [time_intervals_number]
            frequency_intervals_number = [frequency_intervals_number]
        self.time_interval = np.array(time_interval)
        self.frequency_interval = np.array(frequency_interval)
        self.number_time_intervals = np.array(time_intervals_number)
        self.number_frequency_intervals = np.array(frequency_intervals_number)
        self.samples_number = samples_number

        # Error checks
        t_u = 2 * np.pi / (2 * self.number_frequency_intervals * self.frequency_interval)
        if (self.time_interval > t_u).any():
            raise RuntimeError('UQpy: Aliasing might occur during execution')

        self.logger = logging.getLogger(__name__)

        self.random_state = process_random_state(random_state)

        self.samples = None
        self.number_of_variables = None
        self.number_of_dimensions = len(self.number_frequency_intervals)
        self.phi = None

        if self.number_of_dimensions == len(self.power_spectrum.shape):
            self.case = UnivariateStochasticProcess()
        else:
            self.number_of_variables = self.power_spectrum.shape[0]
            self.case = MultivariateStochasticProcess()

        # Run Spectral Representation Method
        if self.samples_number is not None:
            self.run(samples_number=self.samples_number)

    @beartype
    def run(self, samples_number: PositiveInteger):
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
        self.logger.info('UQpy: Stochastic Process: Running Spectral Representation Method.')

        samples = None
        phi = None

        samples = self.case.calculate_samples()

        if self.samples is None:
            self.samples = samples
            self.phi = phi
        else:
            self.samples = np.concatenate((self.samples, samples), axis=0)
            self.phi = np.concatenate((self.phi, phi), axis=0)

        self.logger.info('UQpy: Stochastic Process: Spectral Representation Method Complete.')





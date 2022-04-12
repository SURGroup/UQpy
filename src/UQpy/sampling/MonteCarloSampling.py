from typing import Union
from typing import Optional
from beartype import beartype

from UQpy.utilities.ValidationTypes import RandomStateType, PositiveInteger, NumpyFloatArray
from UQpy.distributions import *
from UQpy.utilities.Utilities import process_random_state
import numpy as np
import logging


class MonteCarloSampling:

    @beartype
    def __init__(
        self,
        distributions: Union[Distribution, list[Distribution]],
        nsamples: Optional[int] = None,
        random_state: RandomStateType = None,
    ):
        """
        Perform Monte Carlo sampling (MCS) of random variables.

        :param distributions: Probability distribution of each random variable.
        :param nsamples: Number of samples to be drawn from each distribution. The :meth:`run` method is
         automatically called if `nsamples` is provided. If `nsamples` is not provided,
         then the :class:`.MonteCarloSampling` object is created but samples are not generated.
        :param random_state: Random seed used to initialize the pseudo-random number generator. If an :any:`int` is
         provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise, the
         object itself can be passed directly.
        """
        self.logger = logging.getLogger(__name__)
        self.random_state = process_random_state(random_state)

        self.list = False
        self.array = False
        self._process_distributions(distributions)

        self.samples: NumpyFloatArray = None
        """Generated samples.
        
        If a list of :class:`.DistributionContinuous1D` objects is provided for `distributions`, then `samples` is an
        :class:`numpy.ndarray` with ``samples.shape=(nsamples, len(distributions))``.
        
        If a :class:`.DistributionContinuous1D` object is provided for `distributions` then `samples` is an array with
        ``samples.shape=(nsamples, 1)``.
        
        If a :class:`.DistributionContinuousND` object is provided for `distributions` then `samples` is an array with
        ``samples.shape=(nsamples, ND)``.
        
        If a list of mixed :class:`.DistributionContinuous1D` and :class:`.DistributionContinuousND` objects is provided
        then `samples` is a list with ``len(samples)=nsamples`` and ``len(samples[i]) = len(distributions)``.
        """
        self.x = None
        self.samplesU01: NumpyFloatArray = None
        """
        Generated samples transformed to the unit hypercube.
        
        This attribute exists only if the :meth:`transform_u01` method is invoked by the user.
        
        If a list of :class:`.DistributionContinuous1D` objects is provided for `distributions`, then `samplesU01` is an
        :class:`numpy.ndarray` with ``samples.shape=(nsamples, len(distributions))``.
        
        If a :class:`.DistributionContinuous1D` object is provided for `distributions` then `samplesU01` is an array 
        with ``samples.shape=(nsamples, 1)``.
        
        If a :class:`.DistributionContinuousND` object is provided for `distributions` then `samplesU01` is an array 
        with ``samples.shape=(nsamples, ND)``.
        
        If a list of mixed :class:`.DistributionContinuous1D` and :class:`.DistributionContinuousND` objects is provided
        then `samplesU01` is a list with ``len(samples)=nsamples`` and ``len(samples[i]) = len(distributions)``.
        """
        self.nsamples = nsamples

        # Run Monte Carlo sampling
        if nsamples is not None:
            self.run(nsamples=self.nsamples, random_state=self.random_state)

    def _process_distributions(self, distributions):
        if isinstance(distributions, list):
            add_continuous_1d = 0
            add_continuous_nd = 0
            for i in range(len(distributions)):
                if not isinstance(distributions[i], Distribution):
                    raise TypeError("UQpy: A UQpy.Distribution object must be provided.")
                if isinstance(distributions[i], DistributionContinuous1D):
                    add_continuous_1d += 1
                elif isinstance(distributions[i], DistributionND):
                    add_continuous_nd += 1
            if add_continuous_1d == len(distributions):
                self.list = False
                self.array = True
            else:
                self.list = True
                self.array = False
            self.dist_object = distributions
        else:
            self.dist_object = distributions
            self.list = False
            self.array = True

    @beartype
    def run(self, nsamples: PositiveInteger, random_state: RandomStateType = None):
        """
        Execute the random sampling in the :class:`.MonteCarloSampling` class.

        The :meth:`run` method is the function that performs random sampling in the :class:`.MonteCarloSampling` class.
        If `nsamples` is provided, the :meth:`run` method is automatically called when the
        :class:`MonteCarloSampling` object is defined. The user may also call the :meth:`run` method directly to
        generate samples. The :meth:`run` method of the :class:`.MonteCarloSampling` class can be  invoked many times
        and each time the generated samples are appended to the existing samples.

        :param nsamples: Number of samples to be drawn from each distribution.

         If the :meth:`run` method is invoked multiple times, the newly generated samples will be appended to the
         existing samples.
        :param random_state: Random seed used to initialize the pseudo-random number generator.
        """
        # Check if a random_state is provided.
        self.random_state = (process_random_state(random_state) if random_state is not None else self.random_state)

        self.logger.info("UQpy: Running Monte Carlo Sampling.")

        if isinstance(self.dist_object, list):
            temp_samples = []
            for i in range(len(self.dist_object)):
                if hasattr(self.dist_object[i], "rvs"):
                    temp_samples.append(self.dist_object[i].rvs(nsamples=nsamples, random_state=self.random_state))
                else:
                    raise ValueError("UQpy: rvs method is missing.")
            self.x = []
            for j in range(nsamples):
                y = [temp_samples[k][j] for k in range(len(self.dist_object))]
                self.x.append(np.array(y))
        elif hasattr(self.dist_object, "rvs"):
            temp_samples = self.dist_object.rvs(nsamples=nsamples, random_state=self.random_state)
            self.x = temp_samples

        if self.samples is None:
            if isinstance(self.dist_object, list) and self.array is True:
                self.samples = np.hstack(np.array(self.x)).T
            else:
                self.samples = np.array(self.x)
        elif isinstance(self.dist_object, list) and self.array is True:
            self.samples = np.concatenate([self.samples, np.hstack(np.array(self.x)).T], axis=0)
        elif isinstance(self.dist_object, Distribution):
            self.samples = np.vstack([self.samples, self.x])
        else:
            self.samples = np.vstack([self.samples, self.x])
        self.nsamples = len(self.samples)

        self.logger.info("UQpy: Monte Carlo Sampling Complete.")

    def transform_u01(self):
        """
        Transform random samples to uniform on the unit hypercube.

        The :meth:`transform_u01` method has no returns, although it creates and/or appends the :py:attr:`samplesU01`
        attribute of the :meth:`.MonteCarloSampling` class.
        """
        if isinstance(self.dist_object, list) and self.array is True:
            zi = np.zeros_like(self.samples)
            for i in range(self.nsamples):
                z = self.samples[i, :]
                for j in range(len(self.dist_object)):
                    if hasattr(self.dist_object[j], "cdf"):
                        zi[i, j] = self.dist_object[j].cdf(z[j])
                    else:
                        raise ValueError("UQpy: All distributions must have a cdf method.")
            self.samplesU01 = zi

        elif isinstance(self.dist_object, Distribution):
            if not hasattr(self.dist_object, "cdf"):
                raise ValueError("UQpy: All distributions must have a cdf method.")

            zi = np.zeros_like(self.samples)
            for i in range(self.nsamples):
                z = self.samples[i, :]
                zi[i, :] = self.dist_object.cdf(z)
            self.samplesU01 = zi
        elif isinstance(self.dist_object, list) and self.list is True:
            temp_samples_u01 = []
            for i in range(self.nsamples):
                z = self.samples[i][:]
                y = [None] * len(self.dist_object)
                for j in range(len(self.dist_object)):
                    if hasattr(self.dist_object[j], "cdf"):
                        zi = self.dist_object[j].cdf(z[j])
                    else:
                        raise ValueError("UQpy: All distributions must have a cdf method.")
                    y[j] = zi
                temp_samples_u01.append(np.array(y))
            self.samplesU01 = temp_samples_u01

import logging
from typing import Union

from beartype import beartype

from UQpy.sampling.stratified_sampling.baseclass.StratifiedSampling import StratifiedSampling
from UQpy.sampling.stratified_sampling.latin_hypercube_criteria.baseclass import Criterion
from UQpy.utilities.Utilities import process_random_state
from UQpy.sampling.stratified_sampling.latin_hypercube_criteria import Random
from UQpy.utilities.ValidationTypes import PositiveInteger, NumpyFloatArray, RandomStateType
from UQpy.distributions import *
import numpy as np
from UQpy.distributions import DistributionContinuous1D, JointIndependent


class LatinHypercubeSampling(StratifiedSampling):
    @beartype
    def __init__(
        self,
        distributions: Union[Distribution, list[Distribution]],
        nsamples: PositiveInteger,
        criterion: Criterion = Random(),
        random_state: RandomStateType = None
    ):
        """
        Perform Latin hypercube sampling (LHS) of random variables.

        All distributions in :class:`LatinHypercubeSampling` must be independent. :class:`.LatinHypercubeSampling` does
        not generate correlated random variables. Therefore, for multi-variate designs the `distributions` must be a
        list of :class:`.DistributionContinuous1D` objects or an object of the :class:`.JointIndependent` class.


        :param distributions: List of :class:`.Distribution` objects
         corresponding to each random variable.
        :param nsamples: Number of samples to be drawn from each distribution.
        :param random_state: Random seed used to initialize the pseudo-random number generator. If an :any:`int` is
         provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise, the
         object itself can be passed directly.
        :param criterion: The criterion for pairing the generating sample points. This parameter must be of
         type :class:`.Criterion`.

         Options:

         1. :class:`.Random` - completely random. \n
         2. :class:`.Centered` - points only at the centre. \n
         3. :class:`.MaxiMin` - maximizing the minimum distance between points. \n
         4. :class:`.MinCorrelation` - minimizing the correlation between the points. \n
         5. User-defined criterion class, by providing an implementation of the abstract class :class:`Criterion`
        """
        self.random_state = process_random_state(random_state)
        self.distributions = distributions
        self.criterion = criterion
        self.nsamples = nsamples
        self.logger = logging.getLogger(__name__)
        self._samples: NumpyFloatArray = None
        if isinstance(self.distributions, list):
            self._samples = np.zeros([self.nsamples, len(self.distributions)])
        elif isinstance(self.distributions, DistributionContinuous1D):
            self._samples = np.zeros([self.nsamples, 1])
        elif isinstance(self.distributions, JointIndependent):
            self._samples = np.zeros([self.nsamples, len(self.distributions.marginals)])

        self.samplesU01: NumpyFloatArray = np.zeros_like(self._samples)
        """The generated LHS samples on the unit hypercube."""

        if self.nsamples is not None:
            self.run(self.nsamples)

    @property
    def samples(self):
        """ The generated LHS samples."""
        return np.atleast_2d(self._samples)

    @beartype
    def run(self, nsamples: PositiveInteger):
        """
        Execute the random sampling in the :class:`.LatinHypercubeSampling` class.

        :param nsamples: If the :meth:`run` method is invoked multiple times, the newly generated samples will
         overwrite the existing samples.

        The :meth:`run` method is the function that performs random sampling in the :class:`.LatinHypercubeSampling`
        class. If `nsamples` is provided, the :meth:`run` method is automatically called when the
        :class:`.LatinHypercubeSampling` object is defined. The user may also call the :meth:`run` method directly to
        generate samples. The :meth:`run` method of the :class:`.LatinHypercubeSampling` class cannot be invoked
        multiple times for sample size extension.

        The :meth:`run` method has no returns, although it creates and/or appends the :py:attr:`samples` and
        :py:attr:`samplesU01` attributes of the :class:`.LatinHypercubeSampling` object.
        """
        self.nsamples = nsamples
        self.logger.info("UQpy: Running Latin Hypercube sampling...")
        self.criterion.create_bins(self._samples, self.random_state)

        u_lhs = self.criterion.generate_samples(self.random_state)
        self.samplesU01 = u_lhs

        if isinstance(self.distributions, list):
            for j in range(len(self.distributions)):
                if hasattr(self.distributions[j], "icdf"):
                    self._samples[:, j] = self.distributions[j].icdf(u_lhs[:, j])

        elif isinstance(self.distributions, JointIndependent):
            if all(hasattr(m, "icdf") for m in self.distributions.marginals):
                for j in range(len(self.distributions.marginals)):
                    self._samples[:, j] = self.distributions.marginals[j].icdf(u_lhs[:, j])

        elif isinstance(self.distributions, DistributionContinuous1D):
            if hasattr(self.distributions, "icdf"):
                self._samples = self.distributions.icdf(u_lhs)

        self.logger.info("Successful execution of LHS design.")

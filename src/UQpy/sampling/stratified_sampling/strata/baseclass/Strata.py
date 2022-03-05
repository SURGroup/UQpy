from typing import Union
import numpy as np
import abc
from beartype import beartype
from UQpy.utilities.ValidationTypes import RandomStateType, NumpyFloatArray
from UQpy.sampling.stratified_sampling.strata.SamplingCriterion import SamplingCriterion


class Strata:
    @beartype
    def __init__(self, seeds: Union[None, np.ndarray] = None, random_state: RandomStateType = None):
        """
        Define a geometric decomposition of the n-dimensional unit hypercube into disjoint and space-filling strata.

        This is the parent class for all spatial stratified_sampling. This parent class only provides the framework for
        stratification and cannot be used directly for the stratification. Stratification is done by calling the child
        class for the desired stratification.

        :param seeds: Define the seed points for the strata. See specific subclass for definition of the seed points.
        """
        self.seeds: NumpyFloatArray = seeds
        """Seed points for the strata. See specific subclass for definition of the seed points."""
        self.volume: NumpyFloatArray = None
        """An array of dimension :code:`(strata_number, )` containing the volume of each stratum. """
        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.Generator object.')

    @abc.abstractmethod
    def stratify(self):
        """
        Perform the stratification of the unit hypercube. It is overwritten by the subclass. This method must exist in
        any subclass of the :class:`.Strata` class.
        """
        pass

    @abc.abstractmethod
    def sample_strata(self, nsamples_per_stratum, random_state):
        """
        Abstract class that need to be implemented in each new Stratum. It defines a way to draw samples from each
        stratum.

        :param nsamples_per_stratum: Number of samples to draw in each stratum
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is :any:`None`.
         If an :any:`int` is provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise,
         the object itself can be passed directly.
        :return: A :class:`tuple` containing the new samples contained in the strata as well as their corresponding
         weights.
        """
        pass

    @abc.abstractmethod
    def calculate_strata_metrics(self, index):
        """
        Abstract method that calculates stratum metrics needed in order for the sampling algorithm to decide which
        stratum to refine

        :param index: Stratum index
        :return: A list containing the metric of each stratum.
        """
        pass

    def initialize(self, samples_number, training_points):
        pass

    def extend_weights(self, samples_per_stratum_number, index, weights):
        if int(samples_per_stratum_number[index]) != 0:
            weights.extend([self.volume[index] / samples_per_stratum_number[index]]
                           * int(samples_per_stratum_number[index]))
        else:
            weights.extend([0] * int(samples_per_stratum_number[index]))

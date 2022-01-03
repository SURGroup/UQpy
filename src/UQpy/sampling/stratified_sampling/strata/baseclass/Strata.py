from typing import Union
import numpy as np
import abc
from beartype import beartype
from UQpy.utilities.ValidationTypes import RandomStateType, NumpyFloatArray
from UQpy.sampling.stratified_sampling.strata.SamplingCriterion import SamplingCriterion


class Strata:
    @beartype
    def __init__(self, seeds: Union[None, np.ndarray] = None, stratification_criterion=SamplingCriterion.RANDOM,
                 random_state: RandomStateType = None):
        """
        Define a geometric decomposition of the n-dimensional unit hypercube into disjoint and space-filling strata.

        This is the parent class for all spatial stratified_sampling. This parent class only provides the framework for
        stratification and cannot be used directly for the stratification. Stratification is done by calling the child
        class for the desired stratification.

        :param seeds: Define the seed points for the strata. See specific subclass for definition of the seed points.
        """
        self.stratification_criterion = stratification_criterion
        self.seeds: NumpyFloatArray = seeds
        """Seed points for the strata. See specific subclass for definition of the seed points."""
        self.volume: NumpyFloatArray = None
        """An array of dimension `(strata_number, )` containing the volume of each stratum. """
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
    def sample_strata(self, samples_per_stratum_number, random_state):
        pass

    @abc.abstractmethod
    def calculate_strata_metrics(self, index):
        pass

    def initialize(self, samples_number, training_points):
        pass

    def extend_weights(self, samples_per_stratum_number, index, weights):
        if int(samples_per_stratum_number[index]) != 0:
            weights.extend(
                [self.volume[index] / samples_per_stratum_number[index]]
                * int(samples_per_stratum_number[index])
            )
        else:
            weights.extend([0] * int(samples_per_stratum_number[index]))

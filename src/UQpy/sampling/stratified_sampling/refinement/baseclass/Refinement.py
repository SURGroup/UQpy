from abc import ABC, abstractmethod
import numpy as np

from UQpy.utilities.ValidationTypes import RandomStateType


class Refinement(ABC):
    """
    Baseclass of all available strata refinement methods. Provides the methods that each existing and new refinement
    algorithm must implement in order to be used in the :class:`.RefinedStratifiedSampling` class.
    """
    @abstractmethod
    def update_samples(
        self,
        nsamples: int,
        samples_per_iteration: int,
        random_state: RandomStateType,
        index: int,
        dimension: int,
        samples_u01: np.ndarray,
        training_points: np.ndarray,
    ):
        """
        Method that need to be overridden in case of new :class:`.Refinement` techniques.

        :param nsamples: Total number of samples to be drawn
        :param samples_per_iteration: New samples to be drawn at each refinement iteration.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is
         :class:`None`.
         If an :any:`int` is provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise,
         the object itself can be passed directly.
        :param index: Iteration index
        :param dimension: Number of dimension of the sampling problem.
        :param samples_u01: Existing samples drawn at the unit hypercube space.
        :param training_points: Training points required in case of advanced refinement techniques.
        """
        pass

    def initialize(self, samples_number, training_points, samples):
        pass

    def finalize(self, samples, samples_per_iteration):
        pass

    @staticmethod
    def identify_bins(strata_metrics, points_to_add, random_state):
        bins2break = np.array([])
        points_left = points_to_add
        while np.where(strata_metrics == strata_metrics.max())[0].shape[0] < points_left:
            bin = np.where(strata_metrics == strata_metrics.max())[0]
            bins2break = np.hstack([bins2break, bin])
            strata_metrics[bin] = 0
            points_left -= bin.shape[0]

        bin_for_remaining_points = random_state.choice(
            np.where(strata_metrics == strata_metrics.max())[0],
            points_left,
            replace=False,)
        bins2break = np.hstack([bins2break, bin_for_remaining_points])
        bins2break = list(map(int, bins2break))
        return bins2break

    @abstractmethod
    def update_strata(self, samplesU01):
        pass


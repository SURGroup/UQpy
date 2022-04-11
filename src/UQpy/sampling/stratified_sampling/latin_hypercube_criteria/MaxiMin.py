import logging

from beartype import beartype

from UQpy.sampling.stratified_sampling.latin_hypercube_criteria import Criterion, Random
from UQpy.utilities.DistanceMetric import DistanceMetric
from scipy.spatial.distance import pdist
import numpy as np
import copy


class MaxiMin(Criterion):
    @beartype
    def __init__(
        self,
        iterations: int = 100,
        metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
    ):
        """
        Method for generating a Latin hypercube design that aims to maximize the minimum sample distance.

        :param iterations: The number of iteration to run in the search for a maximin design.
        :param metric: The distance metric to use. Available options are provided in the :class:`DistanceMetric` enum.
        """
        super().__init__()
        self.iterations = iterations
        self.logger = logging.getLogger(__name__)

        if isinstance(metric, DistanceMetric):
            metric_str = str(metric.name).lower()
            self.distance_function = lambda x: pdist(x, metric=metric_str)
        else:
            raise ValueError("UQpy: Please provide a valid metric.")

        self.random_criterion = Random()

    def create_bins(self, samples, random_state):
        self.random_criterion.create_bins(samples, random_state)
        super().create_bins(samples, random_state)

    def generate_samples(self, random_state):
        i = 0
        lhs_samples = self.random_criterion.generate_samples(random_state)
        distance = self.distance_function(lhs_samples)
        maximized_minimum_distance = np.min(distance)
        while i < self.iterations:
            samples_try = self.random_criterion.generate_samples(random_state)
            distance = self.distance_function(samples_try)
            if maximized_minimum_distance < np.min(distance):
                maximized_minimum_distance = np.min(distance)
                lhs_samples = copy.deepcopy(samples_try)
            i += 1

        self.logger.info("UQpy: Achieved maximum distance of %(distance)s" % {"distance": maximized_minimum_distance})

        return lhs_samples

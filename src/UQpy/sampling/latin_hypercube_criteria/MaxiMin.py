from UQpy.sampling.latin_hypercube_criteria import Criterion, Random
from UQpy.sampling.latin_hypercube_criteria.DistanceMetric import DistanceMetric
from scipy.spatial.distance import pdist
import numpy as np
import copy


class MaxiMin(Criterion):
    """
            Method for generating a Latin hypercube design that aims to maximize the minimum sample distance.

            **Input:**

            * **samples** (`ndarray`):
                A set of samples drawn from within each LHS bin.

            * **random_state** (``numpy.random.RandomState`` object):
                A ``numpy.RandomState`` object that fixes the seed of the pseudo random number generation.

            * **iterations** (`int`):
                The number of iteration to run in the search for a maximin design.

            * **metric** (`str` or `callable`):
                The distance metric to use.
                    Options:
                        1. `str` - Available options are those supported by ``scipy.spatial.distance``
                        2. User-defined function to compute the distance between samples. This function replaces the
                           ``scipy.spatial.distance.pdist`` method.

            **Output/Returns:**

            * **lhs_samples** (`ndarray`)
                The maximin set of LHS samples.

            """
    def __init__(self, samples, random_state=None, iterations=100, metric=DistanceMetric.BRAYCURTIS):
        self.random_state = random_state
        self.samples = samples
        self.iterations = iterations

        if isinstance(metric, str) & isinstance(metric, DistanceMetric):
            self.distance_function = lambda x: pdist(x, metric=metric)
        elif callable(metric):
            self. distance_function = metric
        else:
            raise ValueError("UQpy: Please provide a valid metric.")

        if not isinstance(iterations, int):
            raise ValueError('UQpy: number of iterations must be an integer.')

        self.random_criterion = Random()

    def generate_samples(self):
        i = 0
        lhs_samples = self.random_criterion(self.samples, self.random_state)
        distance = self.distance_function(lhs_samples)
        maximized_minimum_distance = np.min(distance)
        while i < self.iterations:
            samples_try = self.random_criterion(self.samples, self.random_state)
            distance = self.distance_function(samples_try)
            if maximized_minimum_distance < np.min(distance):
                maximized_minimum_distance = np.min(distance)
                lhs_samples = copy.deepcopy(samples_try)
            i = i + 1

        if self.verbose:
            print('UQpy: Achieved maximum distance of ', maximized_minimum_distance)

        return lhs_samples

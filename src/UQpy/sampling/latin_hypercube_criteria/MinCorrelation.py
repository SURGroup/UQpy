import logging

from beartype import beartype

from UQpy.sampling.latin_hypercube_criteria import Criterion, Random
import numpy as np
import copy

from UQpy.utilities.ValidationTypes import RandomStateType


class MinCorrelation(Criterion):
    """
            Method for generating a Latin hypercube design that aims to minimize spurious correlations.

            **Input:**

            * **samples** (`ndarray`):
                A set of samples drawn from within each LHS bin.

            * **random_state** (``numpy.random.RandomState`` object):
                A ``numpy.RandomState`` object that fixes the seed of the pseudo random number generation.

            * **iterations** (`int`):
                The number of iteration to run in the search for a maximin design.

            **Output/Returns:**

            * **lhs_samples** (`ndarray`)
                The minimum correlation set of LHS samples.

            """
    @beartype
    def __init__(self,
                 random_state: RandomStateType = None,
                 iterations: int = 100):
        super.__init__(random_state)

        self.random_state = random_state
        self.iterations = iterations
        self.random_criterion = Random(random_state=random_state)
        self.logger = logging.getLogger(__name__)

    def create_bins(self, samples):
        self.random_criterion.create_bins(samples)
        super().create_bins(samples)

    def generate_samples(self):
        i = 0
        lhs_samples = self.random_criterion.generate_samples()
        r = np.corrcoef(np.transpose(lhs_samples))
        np.fill_diagonal(r, 1)
        r1 = r[r != 1]
        min_corr = np.max(np.abs(r1))
        while i < self.iterations:
            samples_try = self.random_criterion.generate_samples()
            r = np.corrcoef(np.transpose(samples_try))
            np.fill_diagonal(r, 1)
            r1 = r[r != 1]
            if np.max(np.abs(r1)) < min_corr:
                min_corr = np.max(np.abs(r1))
                lhs_samples = copy.deepcopy(samples_try)
            i = i + 1
        self.logger.info('UQpy: Achieved minimum correlation of ', min_corr)

        return lhs_samples

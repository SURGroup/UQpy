import logging
from beartype import beartype
from UQpy.sampling.stratified_sampling.latin_hypercube_criteria import Criterion, Random
import numpy as np
import copy


class MinCorrelation(Criterion):
    @beartype
    def __init__(self, iterations: int = 100):
        """
        Method for generating a Latin hypercube design that aims to minimize spurious correlations.

        :param iterations: The number of iteration to run in the search for a maximin design.
        """
        super().__init__()
        self.iterations = iterations
        self.random_criterion = Random()
        self.logger = logging.getLogger(__name__)

    def create_bins(self, samples, random_state):
        self.random_criterion.create_bins(samples, random_state)
        super().create_bins(samples, random_state)

    def generate_samples(self, random_state):
        i = 0
        lhs_samples = self.random_criterion.generate_samples(random_state)
        r = np.corrcoef(np.transpose(lhs_samples))
        np.fill_diagonal(r, 1)
        r1 = r[r != 1]
        min_corr = np.max(np.abs(r1))
        while i < self.iterations:
            samples_try = self.random_criterion.generate_samples(random_state)
            r = np.corrcoef(np.transpose(samples_try))
            np.fill_diagonal(r, 1)
            r1 = r[r != 1]
            if np.max(np.abs(r1)) < min_corr:
                min_corr = np.max(np.abs(r1))
                lhs_samples = copy.deepcopy(samples_try)
            i += 1
        self.logger.info("UQpy: Achieved minimum correlation of %c", min_corr)

        return lhs_samples

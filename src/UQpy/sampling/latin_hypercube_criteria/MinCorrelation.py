from UQpy.sampling.latin_hypercube_criteria import Criterion, RandomCriterion
import numpy as np
import copy


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

    def __init__(self, samples, random_state=None, iterations=100):
        if not isinstance(iterations, int):
            raise ValueError('UQpy: number of iterations must be an integer.')

        self.samples = samples
        self.random_state = random_state
        self.iterations = iterations
        self.random_criterion = RandomCriterion()

    def generate_samples(self):
        i = 0
        lhs_samples = self.random_criterion(self.samples, self.random_state)
        r = np.corrcoef(np.transpose(lhs_samples))
        np.fill_diagonal(r, 1)
        r1 = r[r != 1]
        min_corr = np.max(np.abs(r1))
        while i < self.iterations:
            samples_try = self.random_criterion(self.samples, self.random_state)
            r = np.corrcoef(np.transpose(samples_try))
            np.fill_diagonal(r, 1)
            r1 = r[r != 1]
            if np.max(np.abs(r1)) < min_corr:
                min_corr = np.max(np.abs(r1))
                lhs_samples = copy.deepcopy(samples_try)
            i = i + 1
        if self.verbose:
            print('UQpy: Achieved minimum correlation of ', min_corr)

        return lhs_samples

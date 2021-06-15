from UQpy.sample_methods.hypercube_criteria import Criterion
import numpy as np


class RandomCriterion(Criterion):
    """
            Method for generating a Latin hypercube design by sampling randomly inside each bin.

            The ``random`` method takes a set of samples drawn randomly from within the Latin hypercube bins and performs a
            random shuffling of them to pair the variables.

            **Input:**

            * **samples** (`ndarray`):
                A set of samples drawn from within each bin.

            * **random_state** (``numpy.random.RandomState`` object):
                A ``numpy.RandomState`` object that fixes the seed of the pseudo random number generation.

            **Output/Returns:**

            * **lhs_samples** (`ndarray`)
                The randomly shuffled set of LHS samples.
            """

    def __init__(self, samples, random_state=None):
        self.samples = samples
        self.random_state = random_state

    def generate_samples(self):
        lhs_samples = np.zeros_like(self.samples)
        samples_number = len(self.samples)
        for j in range(self.samples.shape[1]):
            if self.random_state is not None:
                order = self.random_state.permutation(samples_number)
            else:
                order = np.random.permutation(samples_number)
            lhs_samples[:, j] = self.samples[order, j]

        return lhs_samples

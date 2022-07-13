from UQpy.sampling.stratified_sampling.latin_hypercube_criteria import Criterion
import numpy as np


class Random(Criterion):
    def __init__(self):
        """
        Method for generating a Latin hypercube design by sampling randomly inside each bin.

        The :class:`Random` class takes a set of samples drawn randomly from within the Latin hypercube bins and
        performs a random shuffling of them to pair the variables.

        """
        super().__init__()

    def generate_samples(self, random_state):
        lhs_samples = np.zeros_like(self.samples)
        samples_number = len(self.samples)
        for j in range(self.samples.shape[1]):
            if random_state is not None:
                order = random_state.permutation(samples_number)
            else:
                order = np.random.permutation(samples_number)
            lhs_samples[:, j] = self.samples[order, j]

        return lhs_samples

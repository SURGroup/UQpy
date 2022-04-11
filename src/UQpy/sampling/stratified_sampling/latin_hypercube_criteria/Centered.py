from UQpy.sampling.stratified_sampling.latin_hypercube_criteria import Criterion
import numpy as np


class Centered(Criterion):
    def __init__(self):
        """
        Method for generating a Latin hypercube design with samples centered in the bins.
        """
        super().__init__()

    def generate_samples(self, random_state):
        u_temp = (self.a + self.b) / 2
        lhs_samples = np.zeros([self.samples.shape[0], self.samples.shape[1]])
        for i in range(self.samples.shape[1]):
            if random_state is not None:
                lhs_samples[:, i] = random_state.permutation(u_temp)
            else:
                lhs_samples[:, i] = np.random.permutation(u_temp)

        return lhs_samples

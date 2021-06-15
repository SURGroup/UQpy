from UQpy.sample_methods.hypercube_criteria import Criterion
import numpy as np


class CenteredCriterion(Criterion):
    def __init__(self, samples, random_state=None, a=None, b=None):
        self.samples = samples
        self.random_state = random_state
        self.a = a
        self.b = b

    def generate_samples(self):
        u_temp = (self.a + self.b) / 2
        lhs_samples = np.zeros([self.samples.shape[0], self.samples.shape[1]])
        for i in range(self.samples.shape[1]):
            if self.random_state is not None:
                lhs_samples[:, i] = self.random_state.permutation(u_temp)
            else:
                lhs_samples[:, i] = np.random.permutation(u_temp)

        return lhs_samples

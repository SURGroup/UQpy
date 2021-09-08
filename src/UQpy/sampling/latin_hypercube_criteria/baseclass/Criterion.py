from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as stats
from beartype import beartype

from UQpy.utilities.ValidationTypes import RandomStateType


class Criterion(ABC):

    @beartype
    def __init__(self, random_state: RandomStateType = None):
        self.a = 0
        self.b = 0
        self.samples = np.zeros(shape=(0, 0))
        self.random_state = random_state
        self.random_state = \
            np.random.RandomState(self.random_state) if isinstance(self.random_state, int) else random_state

    def create_bins(self, samples):
        samples_number = samples.shape[0]
        cut = np.linspace(0, 1, samples_number + 1)
        self.a = cut[:samples_number]
        self.b = cut[1:samples_number + 1]

        u = np.zeros(shape=(samples.shape[0], samples.shape[1]))
        self.samples = np.zeros_like(u)
        for i in range(samples.shape[1]):
            u[:, i] = stats.uniform.rvs(size=samples.shape[0],
                                        random_state=self.random_state)
            self.samples[:, i] = u[:, i] * (self.b - self.a) + self.a

    @abstractmethod
    def generate_samples(self):
        pass

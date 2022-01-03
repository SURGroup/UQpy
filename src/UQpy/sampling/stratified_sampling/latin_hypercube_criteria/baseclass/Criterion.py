from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as stats
from beartype import beartype

from UQpy.utilities.ValidationTypes import RandomStateType


class Criterion(ABC):
    @beartype
    def __init__(self):
        self.a = 0
        self.b = 0
        self.samples = np.zeros(shape=(0, 0))

    def create_bins(self, samples, random_state):
        samples_number = samples.shape[0]
        cut = np.linspace(0, 1, samples_number + 1)
        self.a = cut[:samples_number]
        self.b = cut[1: samples_number + 1]

        u = np.zeros(shape=(samples.shape[0], samples.shape[1]))
        self.samples = np.zeros_like(u)
        for i in range(samples.shape[1]):
            u[:, i] = stats.uniform.rvs(size=samples.shape[0], random_state=random_state)
            self.samples[:, i] = u[:, i] * (self.b - self.a) + self.a

    @abstractmethod
    def generate_samples(self, random_state):
        """
        Abstract method that must be overriden when generating creating new Latin Hypercube sampling criteria.
        """
        pass

from abc import ABC, abstractmethod


class Surrogate(ABC):
    @abstractmethod
    def fit(self, samples, values):
        pass

    @abstractmethod
    def predict(self, points, return_std=False):
        pass

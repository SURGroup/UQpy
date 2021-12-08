from abc import ABC, abstractmethod


class InterpolationMethod(ABC):
    @abstractmethod
    def interpolate(self, point):
        pass

    def fit(self, coordinates, manifold_data, samples):
        pass

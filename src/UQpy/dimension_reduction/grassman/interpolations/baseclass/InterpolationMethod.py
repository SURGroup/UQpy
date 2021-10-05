from abc import ABC, abstractmethod


class InterpolationMethod(ABC):

    @abstractmethod
    def interpolate(self, coordinates, samples, point):
        pass

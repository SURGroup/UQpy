from abc import ABC, abstractmethod


class OptimizationMethod(ABC):

    @abstractmethod
    def optimize(self, data_points, distance):
        pass

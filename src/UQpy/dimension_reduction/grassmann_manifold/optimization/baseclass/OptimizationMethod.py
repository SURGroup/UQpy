from abc import ABC, abstractmethod


class OptimizationMethod(ABC):
    """
    The parent class to all  classes used for optimization on the Grassmann manifold.
    """
    @abstractmethod
    def optimize(self, data_points, distance):
        pass

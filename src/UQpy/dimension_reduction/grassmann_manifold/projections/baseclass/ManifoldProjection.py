from abc import ABC, abstractmethod

from UQpy.dimension_reduction.kernels.baseclass.Kernel import Kernel


class ManifoldProjection(ABC):
    """
    The parent class to all classes used to represent data on the Grassmann manifold.
    """
    @abstractmethod
    def evaluate_matrix(self, operator: Kernel):
        pass

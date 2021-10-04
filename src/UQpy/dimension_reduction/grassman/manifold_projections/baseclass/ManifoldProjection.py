from abc import ABC, abstractmethod

from UQpy.dimension_reduction.kernels.grassmanian.baseclass.Kernel import Kernel


class ManifoldProjection(ABC):

    @abstractmethod
    def reconstruct_solution(self, karcher_mean, interpolation, coordinates, point, element_wise=True):
        pass

    @abstractmethod
    def evaluate_matrix(self, operator: Kernel):
        pass

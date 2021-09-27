from abc import ABC, abstractmethod

from UQpy.dimension_reduction.kernels.grassmanian.baseclass.Kernel import Kernel


class ManifoldProjection(ABC):

    @abstractmethod
    def interpolate(self, karcher_mean, interpolator, coordinates, point, element_wise=True):
        pass

    @abstractmethod
    def evaluate_kernel_matrix(self, kernel:Kernel):
        pass

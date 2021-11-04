from abc import ABC, abstractmethod

from UQpy.dimension_reduction.kernels.baseclass.Kernel import Kernel


class ManifoldProjection(ABC):
    """
    A baseclass that binds together different methods used to represent data on the Grassmann manifold.
    """
    @abstractmethod
    def reconstruct_solution(
        self,
        interpolation,
        coordinates,
        point,
        p_planes_dimensions,
        optimization_method,
        distance,
        element_wise=True,
    ):
        pass

    @abstractmethod
    def evaluate_matrix(self, operator: Kernel):
        pass

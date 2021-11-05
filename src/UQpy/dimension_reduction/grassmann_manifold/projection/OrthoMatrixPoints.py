import itertools

from UQpy.utilities.ValidationTypes import Numpy2DFloatArray
from UQpy.dimension_reduction.grassmann_manifold.projection.baseclass.ManifoldProjection import (
    ManifoldProjection,
)
from UQpy.dimension_reduction.kernels.baseclass.Kernel import Kernel
import numpy as np


class OrthoMatrixPoints(ManifoldProjection):
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

    def __init__(self, input_points: list[Numpy2DFloatArray], p_planes_dimensions: int):

        points_number = max(np.shape(input_points[0]))
        for i in range(len(input_points)):
            if points_number != max(np.shape(input_points[i])):
                raise TypeError(
                    "UQpy: The shape of the input matrices must be the same."
                )

        # Check the embedding dimension and its consistency.
        p_dim = []
        for i in range(len(input_points)):
            p_dim.append(min(np.shape(np.array(input_points[i]))))

        if p_dim.count(p_dim[0]) != len(p_dim):
            raise TypeError(
                "UQpy: The input points do not belong to the same manifold."
            )
        else:
            p0 = p_dim[0]
            if p0 != p_planes_dimensions:
                raise ValueError(
                    "UQpy: The input points do not belong to the manifold G(n,p)."
                )

        self.data = input_points
        self.p_planes_dimensions = p0

    def evaluate_matrix(self, kernel: Kernel):
        return kernel.kernel_operator(self.data, self.p_planes_dimensions)

    def interpolate(
        self, karcher_mean, interpolator, coordinates, point, element_wise=True
    ):
        pass

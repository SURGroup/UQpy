from UQpy.dimension_reduction.grassmann_manifold import Grassmann
from UQpy.dimension_reduction.grassmann_manifold.GrassmannPoint import GrassmannPoint
from UQpy.dimension_reduction.grassmann_manifold.interpolation.baseclass import InterpolationMethod
from UQpy.dimension_reduction.grassmann_manifold.optimization.baseclass import OptimizationMethod
from UQpy.utilities.ValidationTypes import NumpyFloatArray


class ManifoldInterpolation:
    def __init__(self, interpolation_method: InterpolationMethod):
        self.interpolation_method = interpolation_method

    def interpolate_manifold(self, manifold_points: list[GrassmannPoint],
                             point_to_interpolate: GrassmannPoint,
                             coordinates: list[NumpyFloatArray],
                             optimization_method: OptimizationMethod, distance):
        mean = Grassmann.karcher_mean(manifold_points=manifold_points,
                                      optimization_method=optimization_method,
                                      distance=distance)

        tangent_points = Grassmann.log_map(manifold_points=manifold_points,
                                           reference_point=mean)

        interp_psi = self.interpolation_method.interpolate(coordinates=coordinates,
                                                           samples=tangent_points,
                                                           point=point_to_interpolate)

        interpolated_point = Grassmann.exp_map(tangent_points=[interp_psi],
                                               reference_point=mean)

        return interpolated_point

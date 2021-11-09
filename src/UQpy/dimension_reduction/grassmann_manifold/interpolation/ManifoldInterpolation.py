from UQpy.dimension_reduction.grassmann_manifold import Grassmann
from UQpy.dimension_reduction.grassmann_manifold.GrassmannPoint import GrassmannPoint
from UQpy.dimension_reduction.grassmann_manifold.interpolation.baseclass import InterpolationMethod
from UQpy.dimension_reduction.grassmann_manifold.optimization.baseclass import OptimizationMethod
from UQpy.utilities.ValidationTypes import NumpyFloatArray
from UQpy.dimension_reduction.distances.baseclass import RiemannianDistance


class ManifoldInterpolation:

    def __init__(self, interpolation_method: InterpolationMethod):
        """
        A class to perform interpolation of points on the Grassmann manifold.

        :param interpolation_method: Type of interpolation.
        """
        self.interpolation_method = interpolation_method

    def interpolate_manifold(self, manifold_data: list[GrassmannPoint],
                             coordinate_to_interpolate: NumpyFloatArray,
                             coordinates: list[NumpyFloatArray],
                             optimization_method: OptimizationMethod, distance: RiemannianDistance):
        """

        :param manifold_data: Data on the Grassmann manifold.
        :param coordinate_to_interpolate:  Point to interpolate.
        :param coordinates: Nodes of the interpolant.
        :param optimization_method: Optimization method for calculating the Karcher mean.
        :param distance:  Distance metric.

        """
        mean = Grassmann.karcher_mean(manifold_points=manifold_data,
                                      optimization_method=optimization_method,
                                      distance=distance)

        tangent_points = Grassmann.log_map(manifold_points=manifold_data,
                                           reference_point=mean)

        interp_psi = self.interpolation_method.interpolate(coordinates=coordinates,
                                                           samples=tangent_points,
                                                           point=coordinate_to_interpolate)

        interpolated_point = Grassmann.exp_map(tangent_points=[interp_psi],
                                               reference_point=mean)

        return interpolated_point

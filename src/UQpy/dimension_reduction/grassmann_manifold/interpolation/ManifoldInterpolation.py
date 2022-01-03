from UQpy.dimension_reduction.grassmann_manifold import Grassmann
from UQpy.dimension_reduction.grassmann_manifold.GrassmannPoint import GrassmannPoint
from UQpy.dimension_reduction.grassmann_manifold.interpolation.baseclass import InterpolationMethod
from UQpy.optimization.baseclass.OptimizationMethod import OptimizationMethod
from UQpy.utilities.ValidationTypes import NumpyFloatArray
from UQpy.dimension_reduction.distances.baseclass import RiemannianDistance


class ManifoldInterpolation:

    def __init__(self, interpolation_method: InterpolationMethod,
                 manifold_data: list[GrassmannPoint],
                 optimization_method: OptimizationMethod,
                 coordinates: list[NumpyFloatArray],
                 distance: RiemannianDistance):
        """
        A class to perform interpolation of points on the Grassmann manifold.

        :param interpolation_method: Type of interpolation.
        :param manifold_data: Data on the Grassmann manifold.
        :param optimization_method: Optimization method for calculating the Karcher mean.
        :param coordinates: Nodes of the interpolant.
        :param distance:  Distance metric.
        """
        self.interpolation_method = interpolation_method

        self.mean = Grassmann.karcher_mean(manifold_points=manifold_data,
                                           optimization_method=optimization_method,
                                           distance=distance)

        self.tangent_points = Grassmann.log_map(manifold_points=manifold_data,
                                                reference_point=self.mean)

        self.interpolation_method.fit(coordinates=coordinates,
                                      manifold_data=manifold_data,
                                      samples=self.tangent_points)

    def interpolate_manifold(self, coordinate_to_interpolate: NumpyFloatArray,):
        """
        :param coordinate_to_interpolate:  Point to interpolate.
        """

        interp_psi = self.interpolation_method.interpolate(point=coordinate_to_interpolate)

        return Grassmann.exp_map(tangent_points=[interp_psi], reference_point=self.mean)

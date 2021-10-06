from UQpy.dimension_reduction.distances.grassmanian.GrassmanDistance import GrassmannDistance
from UQpy.dimension_reduction.grassman.methods.KarcherMean import KarcherMean
from UQpy.dimension_reduction.grassman.interpolations.LinearInterpolation import LinearInterpolation
from UQpy.dimension_reduction.grassman.manifold_projections.baseclass.ManifoldProjection import ManifoldProjection
from UQpy.dimension_reduction.grassman.optimization_methods.GradientDescent import GradientDescent


class Grassmann:

    def __init__(self, manifold_projected_points: ManifoldProjection):
        self.manifold_projected_points = manifold_projected_points

    # def interpolate(self, coordinates, point, element_wise=True,
    #                 karcher_mean: KarcherMean =None,
    #                 interpolation_method=None):
    #     if karcher_mean is None:
    #         karcher_mean = KarcherMean(distance=GrassmannDistance(), optimization_method=GradientDescent(),
    #                                    p_planes_dimensions=self.manifold_projected_points.p_planes_dimensions)
    #     if interpolation_method is None:
    #         interpolation_method = LinearInterpolation()
    #
    #     return self.manifold_projected_points.interpolate(karcher_mean, interpolation_method,
    #                                                       coordinates, point, element_wise)

    def evaluate_kernel_matrix(self, kernel):
        kernel_matrix = self.manifold_projected_points.evaluate_matrix(kernel)
        return kernel_matrix

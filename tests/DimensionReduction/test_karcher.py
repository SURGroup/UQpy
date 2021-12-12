import copy

import numpy as np
from beartype import beartype

from UQpy.dimension_reduction.distances.GeodesicDistance import GeodesicDistance
from UQpy.dimension_reduction.grassmann_manifold.GrassmannPoint import GrassmannPoint
from UQpy.optimization.baseclass.OptimizationMethod import \
    OptimizationMethod
from UQpy.dimension_reduction.grassmann_manifold.projections.SvdProjection import SvdProjection
from UQpy.dimension_reduction.grassmann_manifold.Grassmann import Grassmann
import sys
from UQpy.optimization.GradientDescent import GradientDescent


def test_karcher():
    np.random.seed(1111)
    # Solutions: original space.
    sol0 = np.dot(np.random.rand(6, 2), np.random.rand(2, 6))
    sol1 = np.dot(np.random.rand(6, 3), np.random.rand(3, 6))
    sol2 = np.dot(np.random.rand(6, 4), np.random.rand(4, 6))
    sol3 = np.dot(np.random.rand(6, 3), np.random.rand(3, 6))

    # Creating a list of matrices.
    matrices = [sol0, sol1, sol2, sol3]
    manifold_projection = SvdProjection(matrices, p=sys.maxsize)

    optimization_method = GradientDescent(acceleration=True, error_tolerance=1e-4, max_iterations=1000)
    psi_mean = Grassmann.karcher_mean(manifold_points=manifold_projection.psi,
                                      optimization_method=optimization_method,
                                      distance=GeodesicDistance())

    phi_mean = Grassmann.karcher_mean(manifold_points=manifold_projection.phi,
                                      optimization_method=optimization_method,
                                      distance=GeodesicDistance())

    assert round(psi_mean.data[0, 0], 9) == -0.399231356
    assert round(phi_mean.data[0, 0], 9) == -0.382092372


def test_user_karcher():
    import numpy as np

    D1 = 6
    r0 = 2  # rank sample 0
    r1 = 3  # rank sample 1
    r2 = 4  # rank sample 2
    r3 = 3  # rank sample 2

    np.random.seed(1111)  # For reproducibility.
    # Solutions: original space.
    Sol0 = np.dot(np.random.rand(D1, r0), np.random.rand(r0, D1))
    Sol1 = np.dot(np.random.rand(D1, r1), np.random.rand(r1, D1))
    Sol2 = np.dot(np.random.rand(D1, r2), np.random.rand(r2, D1))
    Sol3 = np.dot(np.random.rand(D1, r3), np.random.rand(r3, D1))

    # Creating a list of matrices.
    matrices = [Sol0, Sol1, Sol2, Sol3]

    class UserOptimizationMethod(OptimizationMethod):

        def __init__(self, acceleration: bool = False,
                     error_tolerance: float = 1e-3,
                     max_iterations: int = 1000):
            self.max_iterations = max_iterations
            self.error_tolerance = error_tolerance
            self.acceleration = acceleration

        def optimize(self, data_points: list[GrassmannPoint], distance):
            # Number of points.
            points_number = len(data_points)
            alpha = 0.5
            rank = []
            for i in range(points_number):
                rank.append(min(np.shape(data_points[i].data)))

            from UQpy.dimension_reduction.grassmann_manifold.Grassmann import Grassmann

            max_rank = max(rank)
            fmean = []
            for i in range(points_number):
                fmean.append(
                    Grassmann.frechet_variance(reference_point=data_points[i], manifold_points=data_points,
                                               distance=distance)
                )

            index_0 = fmean.index(min(fmean))
            mean_element = GrassmannPoint(data_points[index_0].data)

            avg_gamma = np.zeros([np.shape(data_points[0].data)[0], np.shape(data_points[0].data)[1]])

            counter_iteration = 0

            l = 0
            avg = []
            _gamma = []
            if self.acceleration:
                _gamma = Grassmann.log_map(manifold_points=data_points,
                                           reference_point=mean_element)

                avg_gamma.fill(0)
                for i in range(points_number):
                    avg_gamma += _gamma[i] / points_number
                avg.append(avg_gamma)

            # Main loop
            while counter_iteration <= self.max_iterations:
                _gamma = Grassmann.log_map(manifold_points=data_points,
                                           reference_point=mean_element)
                avg_gamma.fill(0)

                for i in range(points_number):
                    avg_gamma += _gamma[i] / points_number

                test_0 = np.linalg.norm(avg_gamma, "fro")
                if test_0 < self.error_tolerance and counter_iteration == 0:
                    break

                # Nesterov: Accelerated Gradient Descent
                if self.acceleration:
                    avg.append(avg_gamma)
                    l0 = l
                    l1 = 0.5 * (1 + np.sqrt(1 + 4 * l * l))
                    ls = (1 - l0) / l1
                    step = (1 - ls) * avg[counter_iteration + 1] + ls * avg[
                        counter_iteration
                    ]
                    l = copy.copy(l1)
                else:
                    step = alpha * avg_gamma

                x = Grassmann.exp_map(tangent_points=[step], reference_point=mean_element)

                test_1 = np.linalg.norm(x[0].data - mean_element.data, "fro")

                if test_1 < self.error_tolerance:
                    break

                mean_element = []
                mean_element = x[0]

                counter_iteration += 1

            # return the Karcher mean.
            return mean_element

    from UQpy.dimension_reduction.distances.baseclass.RiemannianDistance import RiemannianDistance

    class UserDistance(RiemannianDistance):
        @beartype
        def compute_distance(self, xi: GrassmannPoint, xj: GrassmannPoint):
            RiemannianDistance.check_rows(xi, xj)

            rank_i = xi.data.shape[1]
            rank_j = xj.data.shape[1]

            r = np.dot(xi.data.T, xj.data)
            (ui, si, vi) = np.linalg.svd(r, full_matrices=True)
            si[np.where(si > 1)] = 1.0
            theta = np.arccos(si)
            distance = (np.sqrt(abs(rank_i - rank_j) * np.pi ** 2 / 4 + np.sum(theta ** 2)))

            return distance

    manifold_projection = SvdProjection(matrices, p=sys.maxsize)
    psi_mean = Grassmann.karcher_mean(manifold_points=manifold_projection.psi,
                                      optimization_method=GradientDescent(acceleration=True, error_tolerance=1e-4,
                                                                          max_iterations=1000),
                                      distance=GeodesicDistance())

    assert round(psi_mean.data[0, 0], 9) == -0.399231356


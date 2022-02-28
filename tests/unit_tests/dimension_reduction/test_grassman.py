import sys

import numpy as np
import scipy

from UQpy.utilities.distances.grassmannian_distances import GeodesicDistance
from UQpy.dimension_reduction.grassmann_manifold.interpolation.methods.LinearInterpolation import LinearInterpolation
from UQpy.dimension_reduction.grassmann_manifold.interpolation.baseclass.InterpolationMethod import InterpolationMethod
from UQpy.dimension_reduction.grassmann_manifold.projections.SvdProjection import SvdProjection
from UQpy.optimization.GradientDescent import GradientDescent
from UQpy.dimension_reduction.grassmann_manifold.ManifoldInterpolation import ManifoldInterpolation


def test_solution_reconstruction():
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    point = np.array([0.1, 0.1])  # Point to interpolate.

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

    # Creating a list of solutions.
    Solutions = [Sol0, Sol1, Sol2, Sol3]

    manifold_projection = SvdProjection(Solutions, p_planes_dimensions=sys.maxsize)

    optimization_method = GradientDescent(acceleration=False, error_tolerance=1e-3, max_iterations=1000)
    interpolation = ManifoldInterpolation(LinearInterpolation())

    interpolated_solution = manifold_projection\
        .reconstruct_solution(interpolation=interpolation, coordinates=nodes,
                              point=point,
                              p_planes_dimensions=manifold_projection.p_planes_dimensions,
                              optimization_method=optimization_method,
                              distance=GeodesicDistance(),
                              element_wise=False)
    assert round(interpolated_solution[0, 0], 9) == 0.61556849


def test_user_interpolation():
    import numpy as np
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # node_0, node_1, node_2.
    point = np.array([0.1, 0.1])  # Point to interpolate.

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

    # Creating a list of solutions.
    Solutions = [Sol0, Sol1, Sol2, Sol3]

    manifold_projection = SvdProjection(Solutions, p_planes_dimensions=sys.maxsize)

    optimization_method = GradientDescent(acceleration=False, error_tolerance=1e-3, max_iterations=1000)

    class UserInterpolation(InterpolationMethod):

        def interpolate(self, coordinates, samples, point):

            """

            **Input:**

            * **coordinates** (`ndarray`)
                Coordinates of the input data points.

            * **samples** (`ndarray`)
                Matrices corresponding to the points on the Grassmann manifold.

            * **point** (`ndarray`)
                Coordinates of the point to be interpolated.

            **Output/Returns:**

            * **interp_point** (`ndarray`)
                Interpolated point.
            """

            if not isinstance(coordinates, list) and not isinstance(coordinates, np.ndarray):
                raise TypeError('UQpy: `coordinates` must be either list or ndarray.')
            else:
                coordinates = np.array(coordinates)

            if not isinstance(samples, list) and not isinstance(samples, np.ndarray):
                raise TypeError('UQpy: `samples` must be either list or ndarray.')
            else:
                samples = np.array(samples)

            if not isinstance(point, list) and not isinstance(point, np.ndarray):
                raise TypeError('UQpy: `point` must be either list or ndarray.')
            else:
                point = np.array(point)

            myInterpolator = scipy.interpolate.LinearNDInterpolator(coordinates, samples)
            interp_point = myInterpolator(point)
            interp_point = interp_point[0]

            return interp_point

    interpolation = ManifoldInterpolation(UserInterpolation())

    interpolated_solution = manifold_projection.reconstruct_solution(interpolation=interpolation, coordinates=nodes,
                                                                     point=point,
                                                                     p_planes_dimensions=manifold_projection.p_planes_dimensions,
                                                                     optimization_method=optimization_method,
                                                                     distance=GeodesicDistance(),
                                                                     element_wise=False)

    assert round(interpolated_solution[0, 0], 9) == 0.61556849


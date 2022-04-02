import sys

import numpy as np
import scipy

from UQpy.utilities.distances.grassmannian_distances import GeodesicDistance
from UQpy.dimension_reduction.grassmann_manifold.projections.SvdProjection import SvdProjection
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

    manifold_projection = SvdProjection(Solutions, p="max")

    interpolation = ManifoldInterpolation(interpolation_method=None,
                                          manifold_data=manifold_projection.phi,
                                          coordinates=nodes,
                                          distance=GeodesicDistance())

    interpolated_solution = interpolation.interpolate_manifold(point=point)
    assert round(interpolated_solution.data[0, 0], 9) == -0.410315255


def test_parsimonious():
    from UQpy.utilities.kernels.GaussianKernel import GaussianKernel
    from UQpy.dimension_reduction.diffusion_maps.DiffusionMaps import DiffusionMaps
    from sklearn.datasets import make_s_curve

    n = 4000
    X, X_color = make_s_curve(n, random_state=3, noise=0)
    kernel = GaussianKernel()

    dmaps_object = DiffusionMaps.build_from_data(data=X,
                                                 alpha=1.0, n_eigenvectors=9,
                                                 is_sparse=True, n_neighbors=100,
                                                 optimize_parameters=True,
                                                 kernel=kernel)

    dmaps_object.fit()
    index, residuals = DiffusionMaps.parsimonious(dmaps_object.eigenvectors, 2)

    assert index[0] == 1
    assert index[1] == 5


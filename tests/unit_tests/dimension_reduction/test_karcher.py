import copy

import numpy as np
from beartype import beartype

from UQpy.utilities.distances.grassmannian_distances import GeodesicDistance
from UQpy.utilities.GrassmannPoint import GrassmannPoint
from UQpy.dimension_reduction.grassmann_manifold.projections.SvdProjection import SvdProjection
from UQpy.dimension_reduction.grassmann_manifold.Grassmann import Grassmann
import sys


def test_karcher():
    np.random.seed(1111)
    # Solutions: original space.
    sol0 = np.dot(np.random.rand(6, 2), np.random.rand(2, 6))
    sol1 = np.dot(np.random.rand(6, 3), np.random.rand(3, 6))
    sol2 = np.dot(np.random.rand(6, 4), np.random.rand(4, 6))
    sol3 = np.dot(np.random.rand(6, 3), np.random.rand(3, 6))

    # Creating a list of matrices.
    matrices = [sol0, sol1, sol2, sol3]
    manifold_projection = SvdProjection(matrices, p="max")

    # optimization_method = GradientDescent(acceleration=True, error_tolerance=1e-4, max_iterations=1000)
    psi_mean = Grassmann.karcher_mean(grassmann_points=manifold_projection.psi,
                                      optimization_method="GradientDescent",
                                      distance=GeodesicDistance())

    phi_mean = Grassmann.karcher_mean(grassmann_points=manifold_projection.phi,
                                      optimization_method="GradientDescent",
                                      distance=GeodesicDistance())

    assert round(psi_mean.data[0, 0], 9) == -0.398422602
    assert round(phi_mean.data[0, 0], 9) == -0.382608986


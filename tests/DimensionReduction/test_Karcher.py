import numpy as np
from UQpy.dimension_reduction.grassman.manifold_projections.SvdProjection import SvdProjection
from UQpy.dimension_reduction.grassman.Grassman import Grassmann
import sys
from UQpy.dimension_reduction.grassman.optimization_methods.GradientDescent import GradientDescent


def test_karcher():
    np.random.seed(1111)  # For reproducibility.
    # Solutions: original space.
    sol0 = np.dot(np.random.rand(6, 2), np.random.rand(2, 6))
    sol1 = np.dot(np.random.rand(6, 3), np.random.rand(3, 6))
    sol2 = np.dot(np.random.rand(6, 4), np.random.rand(4, 6))
    sol3 = np.dot(np.random.rand(6, 3), np.random.rand(3, 6))

    # Creating a list of matrices.
    matrices = [sol0, sol1, sol2, sol3]
    manifold_projection = SvdProjection(matrices, p_planes_dimensions=sys.maxsize)

    optimization_method = GradientDescent(acceleration=True, error_tolerance=1e-4, max_iterations=1000)
    psi_mean = Grassmann.karcher_mean(manifold_points=manifold_projection.psi,
                                      p_planes_dimensions=manifold_projection.p_planes_dimensions,
                                      optimization_method=optimization_method,
                                      distance=Grassmann())

    phi_mean = Grassmann.karcher_mean(manifold_points=manifold_projection.phi,
                                      p_planes_dimensions=manifold_projection.p_planes_dimensions,
                                      optimization_method=optimization_method,
                                      distance=Grassmann())

    assert psi_mean[0, 0] == -0.3992313564023919
    assert phi_mean[0, 0] == -0.3820923720323338

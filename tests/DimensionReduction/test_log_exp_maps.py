import numpy as np
from UQpy.dimension_reduction.grassmann_manifold.manifold_projections.SvdProjection import SvdProjection
from UQpy.dimension_reduction.grassmann_manifold.Grassman import Grassmann
import sys


def test_log_exp_maps():
    np.random.seed(1111)  # For reproducibility.
    # Solutions: original space.
    sol0 = np.dot(np.random.rand(6, 2), np.random.rand(2, 6))
    sol1 = np.dot(np.random.rand(6, 3), np.random.rand(3, 6))
    sol2 = np.dot(np.random.rand(6, 4), np.random.rand(4, 6))
    sol3 = np.dot(np.random.rand(6, 3), np.random.rand(3, 6))

    # Creating a list of matrices.
    matrices = [sol0, sol1, sol2, sol3]
    manifold_projection = SvdProjection(matrices, p_planes_dimensions=sys.maxsize)

    points_tangent = Grassmann.log_map(manifold_points=manifold_projection.psi,
                                       reference_point=manifold_projection.psi[0])

    assert np.round(points_tangent[0][0][0], 2) == 0.0
    assert points_tangent[1][0][0] == -0.002899719618992682
    assert points_tangent[2][0][0] == 0.012574949291454723
    assert points_tangent[3][0][0] == 0.017116995689638644

    manifold_points = Grassmann.exp_map(tangent_points=points_tangent,
                                        reference_point=manifold_projection.psi[0])

    assert np.round(manifold_points[0][0][0], 5) == -0.49845
    assert manifold_points[1][0][0] == -0.5013644537794977
    assert manifold_points[2][0][0] == -0.4498150778515823
    assert manifold_points[3][0][0] == -0.4980716121312462



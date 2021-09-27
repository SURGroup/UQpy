import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from UQpy.dimension_reduction.distances.grassmanian.GrassmanDistance import GrassmannDistance
from UQpy.dimension_reduction.grassman.Grassman import Grassmann
from UQpy.dimension_reduction.grassman.KarcherMean import KarcherMean
from UQpy.dimension_reduction.grassman.interpolations.LinearInterpolation import LinearInterpolation
from UQpy.dimension_reduction.grassman.manifold_projections.SvdProjection import SvdProjection
from UQpy.dimension_reduction.grassman.optimization_methods.GradientDescent import GradientDescent
from UQpy.dimension_reduction.kernels.grassmanian.ProjectionKernel import ProjectionKernel
from UQpy.dimension_reduction.euclidean.Euclidean import Euclidean

def test_log_exp():
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

    manifold_projection = SvdProjection(matrices, p_planes_dimensions=sys.maxsize)

    points_tangent = Grassmann.log_map(points_grassmann=manifold_projection.psi,
                                       reference_point=manifold_projection.psi[0])

    assert points_tangent[0][0][0] == 0.0
    assert points_tangent[1][0][0] == -0.002899719618992682
    assert points_tangent[2][0][0] == 0.012574949291454723
    assert points_tangent[3][0][0] == 0.017116995689638644

    points_grassmann = Grassmann.exp_map(points_tangent=points_tangent,
                                         reference_point=manifold_projection.psi[0])

    assert points_grassmann[0][0][0] == -0.4984521191998955
    assert points_grassmann[1][0][0] == -0.5013644537794977
    assert points_grassmann[2][0][0] == -0.4498150778515823
    assert points_grassmann[3][0][0] == -0.4980716121312462


def test_karcher():
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

    manifold_projection = SvdProjection(matrices, p_planes_dimensions=sys.maxsize)

    optimization_method = GradientDescent(acceleration=True, error_tolerance=1e-4, max_iterations=1000)
    karcher = KarcherMean(distance=GrassmannDistance(), optimization_method=optimization_method,
                          p_planes_dimensions=manifold_projection.p_planes_dimensions)

    psi_mean = karcher.compute_mean(manifold_projection.psi)
    phi_mean = karcher.compute_mean(manifold_projection.phi)

    assert psi_mean[0, 0] == -0.3992313564023919
    assert phi_mean[0, 0] == -0.3820923720323338


def test_distances():
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

    manifold_projection = SvdProjection(matrices, p_planes_dimensions=sys.maxsize)

    distance_metric = GrassmannDistance()
    value = distance_metric.compute_distance(manifold_projection.psi[0], manifold_projection.psi[1])

    assert value == 5.672445010189097


def test_interpolation():
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

    # optimization_method = GradientDescent(acceleration=False, error_tolerance=1e-3, max_iterations=1000)
    # karcher = KarcherMean(distance=GrassmannDistance(), optimization_method=optimization_method,
    #                       p_planes_dimensions=manifold_projection.p_planes_dimensions)
    # interpolated_solution = manifold_projection.interpolate(karcher_mean=karcher, interpolator=LinearInterpolation(),
    #                                                         coordinates=nodes, point=point, element_wise=False)

    manifold = Grassmann(manifold_projection)
    interpolated_solution = manifold.interpolate(coordinates=nodes, point=point, element_wise=False)
    assert interpolated_solution[0, 0] == 0.6155684900619302


def test_kernel():
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
    manifold = Grassmann(manifold_projected_points=manifold_projection)
    kernel = manifold.evaluate_kernel_matrix(kernel=ProjectionKernel())

    assert kernel[0, 1] == 2.740411439218967



def test_dmaps():
    import numpy as np
    from UQpy.dimension_reduction.DiffusionMaps import DiffusionMaps

    # set parameters
    length_phi = 15  # length of swiss roll in angular direction
    length_Z = 15  # length of swiss roll in z direction
    sigma = 0.1  # noise strength
    m = 4000  # number of samples

    np.random.seed(1111)
    # create dataset
    phi = length_phi * np.random.rand(m)
    xi = np.random.rand(m)
    Z0 = length_Z * np.random.rand(m)
    X0 = 1. / 6 * (phi + sigma * xi) * np.sin(phi)
    Y0 = 1. / 6 * (phi + sigma * xi) * np.cos(phi)

    swiss_roll = np.array([X0, Y0, Z0]).transpose()

    kernel = Euclidean(data=swiss_roll, epsilon=0.03)
    kernel_matrix = kernel.evaluate_kernel_matrix()

    dmaps = DiffusionMaps(alpha=0.5, eigenvectors_number=3,
                          is_sparse=True, neighbors_number=100,
                          kernel_matrix=kernel_matrix)

    diff_coords, evals, evecs = dmaps.mapping()

    assert evals[0] == 1.0000000000000016
    assert evals[1] == 0.9994892947282611
    assert evals[2] == 0.999116765777085

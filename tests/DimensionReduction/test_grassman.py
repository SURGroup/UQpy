import copy
import sys

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from UQpy.dimension_reduction.distances.grassmanian.GrassmannDistance import Grassmann
from UQpy.dimension_reduction.distances.grassmanian.baseclass.RiemannianDistance import RiemannianDistance
from UQpy.dimension_reduction.grassman.Grassman import Grassmann
from UQpy.dimension_reduction.grassman.interpolations.LinearInterpolation import LinearInterpolation
from UQpy.dimension_reduction.grassman.interpolations.baseclass.InterpolationMethod import InterpolationMethod
from UQpy.dimension_reduction.grassman.manifold_projections.SvdProjection import SvdProjection
from UQpy.dimension_reduction.grassman.optimization_methods.GradientDescent import GradientDescent
from UQpy.dimension_reduction.grassman.optimization_methods.baseclass.OptimizationMethod import OptimizationMethod
from UQpy.dimension_reduction.kernels.grassmanian.ProjectionKernel import ProjectionKernel
from UQpy.dimension_reduction.grassman.interpolations.Interpolation import Interpolation

from UQpy.dimension_reduction.DiffusionMaps import DiffusionMaps
from UQpy.dimension_reduction.kernels.baseclass.Kernel import Kernel
from UQpy.dimension_reduction.kernels.euclidean.GaussianKernel import GaussianKernel


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

    points_tangent = Grassmann.log_map(manifold_points=manifold_projection.psi,
                                       reference_point=manifold_projection.psi[0])

    assert points_tangent[0][0][0] == 0.0
    assert points_tangent[1][0][0] == -0.002899719618992682
    assert points_tangent[2][0][0] == 0.012574949291454723
    assert points_tangent[3][0][0] == 0.017116995689638644

    points_grassmann = Grassmann.exp_map(tangent_points=points_tangent,
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
    psi_mean = Grassmann.karcher_mean(points_grassmann=manifold_projection.psi,
                                      p_planes_dimensions=manifold_projection.p_planes_dimensions,
                                      optimization_method=optimization_method,
                                      distance=Grassmann())

    phi_mean = Grassmann.karcher_mean(points_grassmann=manifold_projection.phi,
                                      p_planes_dimensions=manifold_projection.p_planes_dimensions,
                                      optimization_method=optimization_method,
                                      distance=Grassmann())

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

    distance_metric = Grassmann()
    value = distance_metric.compute_distance(manifold_projection.psi[0], manifold_projection.psi[1])

    assert value == 5.672445010189097


def test_solution_reconstruction():
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
    interpolation = Interpolation(LinearInterpolation())

    interpolated_solution = manifold_projection.reconstruct_solution(interpolation=interpolation, coordinates=nodes,
                                                                     point=point,
                                                                     p_planes_dimensions=manifold_projection.p_planes_dimensions,
                                                                     optimization_method=optimization_method,
                                                                     distance=Grassmann(),
                                                                     element_wise=False)
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


def test_dmaps_swiss_roll():
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
    dmaps = DiffusionMaps.create_from_data(data=swiss_roll,
                                           alpha=0.5, eigenvectors_number=3,
                                           is_sparse=True, neighbors_number=100,
                                           kernel=GaussianKernel(epsilon=0.03))

    diff_coords, evals, evecs = dmaps.mapping()

    assert evals[0] == 1.0000000000000016
    assert evals[1] == 0.9994892947282611
    assert evals[2] == 0.999116765777085


def test_dmaps_circular():
    import numpy as np
    np.random.seed(1111)
    a = 6
    b = 1
    k = 10
    u = np.linspace(0, 2 * np.pi, 1000)

    v = k * u

    x0 = (a + b * np.cos(0.8 * v)) * (np.cos(u))
    y0 = (a + b * np.cos(0.8 * v)) * (np.sin(u))
    z0 = b * np.sin(0.8 * v)

    rox = 0.2
    roy = 0.2
    roz = 0.2
    x = x0 + rox * np.random.normal(0, 1, len(x0))
    y = y0 + roy * np.random.normal(0, 1, len(y0))
    z = z0 + roz * np.random.normal(0, 1, len(z0))

    X = np.array([x, y, z]).transpose()

    dmaps = DiffusionMaps.create_from_data(data=X, alpha=1, eigenvectors_number=3, kernel=Gaussian(epsilon=0.3))

    diff_coords, evals, evecs = dmaps.mapping()

    assert evals[0] == 1.0000000000000002
    assert evals[1] == 0.9964842223723996
    assert evals[2] == 0.9964453129642372


def test_diff_matrices():
    import numpy as np

    np.random.seed(111)
    npts = 1000
    pts = np.random.rand(npts, 2)

    a0 = 0
    a1 = 1
    b0 = 0
    b1 = 1

    nodes = np.zeros(np.shape(pts))

    nodes[:, 0] = pts[:, 0] * (a1 - a0) + a0
    nodes[:, 1] = pts[:, 1] * (b1 - b0) + b0

    ns = 40

    x = np.linspace(0, 1, ns)
    samples = []
    for i in range(npts):

        M = np.zeros((ns, ns))
        for k in range(ns):
            f = np.sin(0.1 * k * np.pi * nodes[i, 0] * x + 2 * np.pi * nodes[i, 1])
            M[:, k] = f

        samples.append(M)

    dmaps = DiffusionMaps.create_from_data(data=samples, alpha=0.5, eigenvectors_number=10)

    diff_coords, evals, evecs = dmaps.mapping()

    assert evals[0] == 1.0000000000000004
    assert evals[1] == 0.12956887787384186
    assert evals[2] == 0.07277038589085978


def test_grassman_kernel():
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

    # Creating a list of solutions.
    matrices = [Sol0, Sol1, Sol2, Sol3]

    manifold_projection = SvdProjection(matrices, p_planes_dimensions=sys.maxsize)
    manifold = Grassmann(manifold_projected_points=manifold_projection)
    kernel = manifold.evaluate_kernel_matrix(kernel=ProjectionKernel())

    class UserKernel(Kernel):

        def apply_method(self, data):
            data.evaluate_matrix(self, self.kernel_operator)

        def pointwise_operator(self, point1, point2):
            """
                            User defined kernel.

                            **Input:**

                            * **x0** (`list` or `ndarray`)
                                Point on the Grassmann manifold.

                            * **x1** (`list` or `ndarray`)
                                Point on the Grassmann manifold.

                            **Output/Returns:**

                            * **distance** (`float`)
                                Kernel value for x0 and x1.
                            """

            if not isinstance(point1, list) and not isinstance(point1, np.ndarray):
                raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
            else:
                point1 = np.array(point1)

            if not isinstance(point2, list) and not isinstance(point2, np.ndarray):
                raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
            else:
                point2 = np.array(point2)

            r = np.dot(point1.T, point2)
            det = np.linalg.det(r)
            ker = det * det
            return ker

    kernel_user_psi = manifold.evaluate_kernel_matrix(kernel=UserKernel())

    assert kernel_user_psi[0, 0] == 1.0000000000000049


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

            myInterpolator = LinearNDInterpolator(coordinates, samples)
            interp_point = myInterpolator(point)
            interp_point = interp_point[0]

            return interp_point

    interpolation = Interpolation(UserInterpolation())

    interpolated_solution = manifold_projection.reconstruct_solution(interpolation=interpolation, coordinates=nodes,
                                                                     point=point,
                                                                     p_planes_dimensions=manifold_projection.p_planes_dimensions,
                                                                     optimization_method=optimization_method,
                                                                     distance=GrassmannDistance(),
                                                                     element_wise=False)

    assert interpolated_solution[0, 0] == 0.6155684900619302


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

        def optimize(self, data_points, distance):
            # Number of points.
            points_number = len(data_points)
            alpha = 0.5
            rank = []
            for i in range(points_number):
                rank.append(min(np.shape(data_points[i])))

            from UQpy.dimension_reduction.grassman.Grassman import Grassmann
            max_rank = max(rank)
            fmean = []
            for i in range(points_number):
                fmean.append(Grassmann.frechet_variance(data_points[i], data_points, distance))

            index_0 = fmean.index(min(fmean))
            mean_element = data_points[index_0].tolist()

            avg_gamma = np.zeros([np.shape(data_points[0])[0], np.shape(data_points[0])[1]])

            counter_iteration = 0

            l = 0
            avg = []
            _gamma = []
            from UQpy.dimension_reduction.grassman.Grassman import Grassmann
            if self.acceleration:
                _gamma = Grassmann.log_map(points_grassmann=data_points,
                                           reference_point=np.asarray(mean_element))

                avg_gamma.fill(0)
                for i in range(points_number):
                    avg_gamma += _gamma[i] / points_number
                avg.append(avg_gamma)

            # Main loop
            while counter_iteration <= self.max_iterations:
                _gamma = Grassmann.log_map(points_grassmann=data_points,
                                           reference_point=np.asarray(mean_element))
                avg_gamma.fill(0)

                for i in range(points_number):
                    avg_gamma += _gamma[i] / points_number

                test_0 = np.linalg.norm(avg_gamma, 'fro')
                if test_0 < self.error_tolerance and counter_iteration == 0:
                    break

                # Nesterov: Accelerated Gradient Descent
                if self.acceleration:
                    avg.append(avg_gamma)
                    l0 = l
                    l1 = 0.5 * (1 + np.sqrt(1 + 4 * l * l))
                    ls = (1 - l0) / l1
                    step = (1 - ls) * avg[counter_iteration + 1] + ls * avg[counter_iteration]
                    l = copy.copy(l1)
                else:
                    step = alpha * avg_gamma

                x = Grassmann.exp_map(points_tangent=[step], reference_point=np.asarray(mean_element))

                test_1 = np.linalg.norm(x[0] - mean_element, 'fro')

                if test_1 < self.error_tolerance:
                    break

                mean_element = []
                mean_element = x[0]

                counter_iteration += 1

            # return the Karcher mean.
            return mean_element

    class UserDistance(RiemannianDistance):

        def compute_distance(self, point1, point2):
            """
                Estimate the user distance.

                **Input:**

                * **x0** (`list` or `ndarray`)
                    Point on the Grassmann manifold.

                * **x1** (`list` or `ndarray`)
                    Point on the Grassmann manifold.

                **Output/Returns:**

                * **distance** (`float`)
                    Procrustes distance between x0 and x1.
                """

            if not isinstance(point1, list) and not isinstance(point1, np.ndarray):
                raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
            else:
                x0 = np.array(point1)

            if not isinstance(point2, list) and not isinstance(point2, np.ndarray):
                raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
            else:
                x1 = np.array(point2)

            l = min(np.shape(x0))
            k = min(np.shape(x1))
            rank = min(l, k)

            r = np.dot(x0.T, x1)
            # (ui, si, vi) = svd(r, rank)

            ui, si, vi = np.linalg.svd(r, full_matrices=True, hermitian=False)  # Compute the SVD of matrix
            si = np.diag(si)  # Transform the array si into a diagonal matrix containing the singular values
            vi = vi.T  # Transpose of vi

            u = ui[:, :rank]
            s = si[:rank, :rank]
            v = vi[:, :rank]

            index = np.where(si > 1)
            si[index] = 1.0
            theta = np.arccos(si)
            theta = np.sin(theta / 2) ** 2
            distance = np.sqrt(abs(k - l) + 2 * np.sum(theta))

            return distance

    manifold_projection = SvdProjection(matrices, p_planes_dimensions=sys.maxsize)
    psi_mean = Grassmann.karcher_mean(points_grassmann=manifold_projection.psi,
                                      p_planes_dimensions=manifold_projection.p_planes_dimensions,
                                      optimization_method=GradientDescent(acceleration=True, error_tolerance=1e-4,
                                                                          max_iterations=1000),
                                      distance=GrassmannDistance())

    assert psi_mean[0, 0] == -0.3992313564023919

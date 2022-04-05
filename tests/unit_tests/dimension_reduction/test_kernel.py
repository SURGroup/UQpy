import sys

from UQpy.utilities.GrassmannPoint import GrassmannPoint
from UQpy.dimension_reduction.grassmann_manifold.projections.SvdProjection import SvdProjection
from UQpy.utilities.kernels.ProjectionKernel import ProjectionKernel
from UQpy.utilities.kernels.BinetCauchyKernel import BinetCauchyKernel
from UQpy.utilities.kernels.GaussianKernel import GaussianKernel
import numpy as np


def test_kernel_projection():
    xi = GrassmannPoint(np.array([[-np.sqrt(2) / 2, -np.sqrt(2) / 4], [np.sqrt(2) / 2,
                                                                       -np.sqrt(2) / 4], [0, -np.sqrt(3) / 2]]))
    xj = GrassmannPoint(np.array([[0, np.sqrt(2) / 2], [1, 0], [0, -np.sqrt(2) / 2]]))
    xk = GrassmannPoint(np.array([[-0.69535592, -0.0546034], [-0.34016974, -0.85332868],
                                  [-0.63305978, 0.51850616]]))
    points = [xi, xj, xk]
    kernel = np.matrix.round(ProjectionKernel().calculate_kernel_matrix(points), 4)

    assert np.allclose(kernel, np.array([[2, 1.0063, 1.2345], [1.0063, 2, 1.0101], [1.2345, 1.0101, 2]]))


def test_kernel_binet_cauchy():
    xi = GrassmannPoint(np.array([[-np.sqrt(2) / 2, -np.sqrt(2) / 4], [np.sqrt(2) / 2, -np.sqrt(2) / 4],
                                  [0, -np.sqrt(3) / 2]]))
    xj = GrassmannPoint(np.array([[0, np.sqrt(2) / 2], [1, 0], [0, -np.sqrt(2) / 2]]))
    xk = GrassmannPoint(np.array([[-0.69535592, -0.0546034], [-0.34016974, -0.85332868], [-0.63305978, 0.51850616]]))
    points = [xi, xj, xk]
    kernel = np.matrix.round(BinetCauchyKernel().calculate_kernel_matrix(points), 4)

    assert np.allclose(kernel, np.array([[1, 0.0063, 0.2345], [0.0063, 1, 0.0101], [0.2345, 0.0101, 1]]))


def test_kernel_gaussian_1d():
    xi = np.array([1, 2, 3, 4])
    xj = np.array([0.2, -1, 3, 5])
    xk = np.array([1, 2, 3, 4])
    points = [xi, xj, xk]
    gaussian = GaussianKernel(epsilon=2.0)
    kernel = gaussian.calculate_kernel_matrix(points)

    assert np.allclose(np.matrix.round(kernel, 4), np.array([[1., 0.2645, 1.], [0.2645, 1., 0.2645], [1, 0.2645, 1]]),
                       atol=1e-04)
    assert np.round(gaussian.epsilon, 4) == 2


def test_kernel_gaussian_2d():
    xi = np.array([[-np.sqrt(2) / 2, -np.sqrt(2) / 4], [np.sqrt(2) / 2, -np.sqrt(2) / 4], [0, -np.sqrt(3) / 2]])
    xj = np.array([[0, np.sqrt(2) / 2], [1, 0], [0, -np.sqrt(2) / 2]])
    xk = np.array([[-0.69535592, -0.0546034], [-0.34016974, -0.85332868], [-0.63305978, 0.51850616]])
    points = [xi, xj, xk]
    gaussian = GaussianKernel()
    kernel = gaussian.calculate_kernel_matrix(points)

    assert np.allclose(np.matrix.round(kernel, 4), np.array([[1., 0.628, 0.3912],
                                                             [0.628, 1., 0.2534],
                                                             [0.3912, 0.2534, 1.]]))
    assert np.round(gaussian.epsilon, 4) == 1.0


sol0 = np.array([[0.61415, 1.03029, 1.02001, 0.57327, 0.79874, 0.73274],
                 [0.56924, 0.91700, 0.88841, 0.53737, 0.68676, 0.67751],
                 [0.51514, 0.87898, 0.87779, 0.47850, 0.69085, 0.61525],
                 [0.63038, 1.10822, 1.12313, 0.58038, 0.89142, 0.75429],
                 [0.69666, 1.03114, 0.95037, 0.67211, 0.71184, 0.82522],
                 [0.66595, 1.03789, 0.98690, 0.63420, 0.75416, 0.79110]])

sol1 = np.array([[1.05134, 1.37652, 0.95634, 0.85630, 0.47570, 1.22488],
                 [0.16370, 0.63105, 0.14533, 0.81030, 0.44559, 0.43358],
                 [1.23478, 2.10342, 1.04698, 1.68755, 0.92792, 1.73277],
                 [0.90538, 1.64067, 0.62027, 1.17577, 0.63644, 1.34925],
                 [0.58210, 0.75795, 0.65519, 0.65712, 0.37251, 0.65740],
                 [0.99174, 1.59375, 0.63724, 0.89107, 0.47631, 1.36581]])

sol2 = np.array([[1.04142, 0.91670, 1.47962, 1.23350, 0.94111, 0.61858],
                 [1.00464, 0.65684, 1.35136, 1.11288, 0.96093, 0.42340],
                 [1.05567, 1.33192, 1.56286, 1.43412, 0.77044, 0.97182],
                 [0.89812, 0.86136, 1.20204, 1.17892, 0.83788, 0.61160],
                 [0.46935, 0.39371, 0.63534, 0.57856, 0.47615, 0.26407],
                 [1.14102, 0.80869, 1.39123, 1.33076, 0.47719, 0.68170]])

sol3 = np.array([[0.60547, 0.11492, 0.78956, 0.13796, 0.76685, 0.41661],
                 [0.32771, 0.11606, 0.67630, 0.15208, 0.44845, 0.34840],
                 [0.58959, 0.10156, 0.72623, 0.11859, 0.73671, 0.38714],
                 [0.36283, 0.07979, 0.52824, 0.09760, 0.46313, 0.27906],
                 [0.87487, 0.22452, 1.30208, 0.30189, 1.22015, 0.62918],
                 [0.56006, 0.16879, 1.09635, 0.20431, 0.69439, 0.60317]])


def test_kernel():
    D1 = 6
    r0 = 2  # rank sample 0
    r1 = 3  # rank sample 1
    r2 = 4  # rank sample 2
    r3 = 3  # rank sample 2

    np.random.seed(1111)  # For reproducibility.
    from numpy.random import RandomState
    rnd = RandomState(0)

    # Creating a list of solutions.
    Solutions = [sol0, sol1, sol2, sol3]
    from UQpy.dimension_reduction.grassmann_manifold.GrassmannOperations import GrassmannOperations
    manifold_projection = SvdProjection(Solutions, p="max")
    kernel = ProjectionKernel()

    kernel = kernel.calculate_kernel_matrix(manifold_projection.u)

    assert np.round(kernel[0, 1], 8) == 6.0

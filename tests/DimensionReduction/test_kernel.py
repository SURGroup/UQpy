import sys

from UQpy import SvdProjection, Grassmann
from UQpy.dimension_reduction.kernels.grassmanian import ProjectionKernel, BinetCauchyKernel
from UQpy.dimension_reduction.kernels.euclidean import GaussianKernel

import numpy as np


def test_kernel_projection():
    xi = np.array([[-np.sqrt(2) / 2, -np.sqrt(2) / 4], [np.sqrt(2) / 2, -np.sqrt(2) / 4], [0, -np.sqrt(3) / 2]])
    xj = np.array([[0, np.sqrt(2) / 2], [1, 0], [0, -np.sqrt(2) / 2]])
    xk = np.array([[-0.69535592, -0.0546034], [-0.34016974, -0.85332868], [-0.63305978, 0.51850616]])
    points = [xi, xj, xk]
    kernel = np.matrix.round(ProjectionKernel().kernel_operator(points), 4)

    assert np.allclose(kernel, np.array([[2, 1.0063, 1.2345], [1.0063, 2, 1.0101], [1.2345, 1.0101, 2]]))


def test_kernel_binet_cauchy():
    xi = np.array([[-np.sqrt(2) / 2, -np.sqrt(2) / 4], [np.sqrt(2) / 2, -np.sqrt(2) / 4], [0, -np.sqrt(3) / 2]])
    xj = np.array([[0, np.sqrt(2) / 2], [1, 0], [0, -np.sqrt(2) / 2]])
    xk = np.array([[-0.69535592, -0.0546034], [-0.34016974, -0.85332868], [-0.63305978, 0.51850616]])
    points = [xi, xj, xk]
    kernel = np.matrix.round(BinetCauchy().kernel_operator(points), 4)

    assert np.allclose(kernel, np.array([[1, 0.0063, 0.2345], [0.0063, 1, 0.0101], [0.2345, 0.0101, 1]]))


def test_kernel_gaussian_1d():
    xi = np.array([1, 2, 3, 4])
    xj = np.array([0.2, -1, 3, 5])
    xk = np.array([1, 2, 3, 4])
    points = [xi, xj, xk]
    gaussian = GaussianKernel(epsilon=2.0)
    kernel = gaussian.kernel_operator(points)

    assert np.allclose(np.matrix.round(kernel, 4), np.array([[1., 0.2645, 1.], [0.2645, 1., 0.2645], [1, 0.2645, 1]]),
                       atol=1e-04)
    assert np.round(gaussian.epsilon, 4) == 2


def test_kernel_gaussian_2d():
    xi = np.array([[-np.sqrt(2) / 2, -np.sqrt(2) / 4], [np.sqrt(2) / 2, -np.sqrt(2) / 4], [0, -np.sqrt(3) / 2]])
    xj = np.array([[0, np.sqrt(2) / 2], [1, 0], [0, -np.sqrt(2) / 2]])
    xk = np.array([[-0.69535592, -0.0546034], [-0.34016974, -0.85332868], [-0.63305978, 0.51850616]])
    points = [xi, xj, xk]
    gaussian = GaussianKernel()
    kernel = gaussian.kernel_operator(points)

    assert np.allclose(np.matrix.round(kernel, 4), np.array([[1., 0.8834, 0.7788], [0.8834, 1., 0.6937], [0.7788, 0.6937, 1.]]))
    assert np.round(gaussian.epsilon, 4) == 3.7538


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


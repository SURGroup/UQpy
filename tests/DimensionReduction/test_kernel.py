from UQpy.dimension_reduction.kernels.grassmanian import ProjectionKernel, BinetCauchy
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
    kernel, epsilon = GaussianKernel().kernel_operator(points, epsilon=2)

    assert np.allclose(np.matrix.round(kernel, 4), np.array([[1., 0.2645, 1.], [0.2645, 1., 0.2645], [1, 0.2645, 1]]),
                       atol=1e-04)
    assert np.round(epsilon, 4) == 2


def test_kernel_gaussian_2d():
    xi = np.array([[-np.sqrt(2) / 2, -np.sqrt(2) / 4], [np.sqrt(2) / 2, -np.sqrt(2) / 4], [0, -np.sqrt(3) / 2]])
    xj = np.array([[0, np.sqrt(2) / 2], [1, 0], [0, -np.sqrt(2) / 2]])
    xk = np.array([[-0.69535592, -0.0546034], [-0.34016974, -0.85332868], [-0.63305978, 0.51850616]])
    points = [xi, xj, xk]
    kernel, epsilon = GaussianKernel().kernel_operator(points)

    assert np.allclose(np.matrix.round(kernel, 4), np.array([[1., 0.8834, 0.7788], [0.8834, 1., 0.6937], [0.7788, 0.6937, 1.]]))
    assert np.round(epsilon, 4) == 3.7538


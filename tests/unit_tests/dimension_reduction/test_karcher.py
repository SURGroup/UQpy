import numpy as np
from UQpy.utilities.GrassmannPoint import GrassmannPoint
from UQpy.dimension_reduction.grassmann_manifold.GrassmannOperations import GrassmannOperations
from UQpy.utilities.distances.grassmannian_distances.GeodesicDistance import GeodesicDistance


def test_karcher_mean_2point():
    """Test karcher mean on 2 points in the manifold Gr(1, 2)"""
    theta1 = np.deg2rad(60)
    x1 = np.array([[np.cos(theta1)],
                   [np.sin(theta1)]])
    theta2 = np.deg2rad(120)
    x2 = np.array([[np.cos(theta2)],
                   [np.sin(theta2)]])
    points = [GrassmannPoint(x1), GrassmannPoint(x2)]
    mean = GrassmannOperations.karcher_mean(grassmann_points=points,
                                            optimization_method='GradientDescent',
                                            distance=GeodesicDistance(),
                                            tolerance=1e-9)
    theta_solution = np.deg2rad(90)
    solution = np.array([[np.cos(theta_solution)],
                         [np.sin(theta_solution)]])
    assert np.allclose(mean.data, solution)


def test_karcher_mean_3point():
    """Test karcher mean on 3 points in the manifold Gr(1, 2)"""
    points = [np.array([[np.cos(np.deg2rad(theta))],
                        [np.sin(np.deg2rad(theta))]]) for theta in (0, 15, 75, 90)]
    mean = GrassmannOperations.karcher_mean(grassmann_points=points,
                                            optimization_method='GradientDescent',
                                            distance=GeodesicDistance(),
                                            tolerance=1e-9)
    solution = np.array([[np.cos(np.deg2rad(45))],
                         [np.sin(np.deg2rad(45))]])
    assert np.allclose(mean.data, solution)

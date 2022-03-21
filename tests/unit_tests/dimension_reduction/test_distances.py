from UQpy.utilities.GrassmannPoint import GrassmannPoint
from UQpy.dimension_reduction.grassmann_manifold.projections.SvdProjection import SvdProjection
from UQpy.utilities.distances.euclidean_distances import EuclideanDistance
from UQpy.utilities.distances.grassmannian_distances import AsimovDistance, BinetCauchyDistance, FubiniStudyDistance, \
    GeodesicDistance, ProcrustesDistance, ProjectionDistance, SpectralDistance
from UQpy.utilities.distances import MartinDistance
from UQpy.utilities.DistanceMetric import DistanceMetric
import numpy as np
import sys


def test_euclidean_distance_2points():
    x = np.array([[2.0, 3.1], [4.0, 1.25]])
    d = EuclideanDistance(metric=DistanceMetric.EUCLIDEAN)
    distance = np.matrix.round(d.compute_distance(x), 3)

    assert distance == 2.724


def test_euclidean_distance_3points():
    x = np.array([[2.0, 3.1], [2.0, 2.1], [4.0, 1.25]])
    d = EuclideanDistance(metric=DistanceMetric.EUCLIDEAN)
    distance = np.matrix.round(d.compute_distance(x), 3)

    assert np.allclose(distance, np.array([1, 2.724, 2.173]))


def test_grassmann_distance():
    xi = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/4], [np.sqrt(2)/2, -np.sqrt(2)/4], [0, -np.sqrt(3)/2]])
    xj = np.array([[0, np.sqrt(2)/2], [1, 0], [0, -np.sqrt(2)/2]])
    distance = np.round(GeodesicDistance().compute_distance(GrassmannPoint(xi), GrassmannPoint(xj)), 6)
    assert distance == 1.491253


def test_fs_distance():
    xi = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/4], [np.sqrt(2)/2, -np.sqrt(2)/4], [0, -np.sqrt(3)/2]])
    xj = np.array([[0, np.sqrt(2)/2], [1, 0], [0, -np.sqrt(2)/2]])
    distance = np.round(FubiniStudyDistance().compute_distance(GrassmannPoint(xi), GrassmannPoint(xj)), 6)
    assert distance == 1.491253


def test_procrustes_distance():
    xi = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/4], [np.sqrt(2)/2, -np.sqrt(2)/4], [0, -np.sqrt(3)/2]])
    xj = np.array([[0, np.sqrt(2)/2], [1, 0], [0, -np.sqrt(2)/2]])
    distance = np.round(ProcrustesDistance().compute_distance(GrassmannPoint(xi), GrassmannPoint(xj)), 6)
    assert distance == 1.356865


def test_projection_distance():
    xi = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/4], [np.sqrt(2)/2, -np.sqrt(2)/4], [0, -np.sqrt(3)/2]])
    xj = np.array([[0, np.sqrt(2)/2], [1, 0], [0, -np.sqrt(2)/2]])
    distance = np.round(ProjectionDistance().compute_distance(GrassmannPoint(xi), GrassmannPoint(xj)), 6)
    assert distance == 0.996838


def test_binet_cauchy_distance():
    xi = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/4], [np.sqrt(2)/2, -np.sqrt(2)/4], [0, -np.sqrt(3)/2]])
    xj = np.array([[0, np.sqrt(2)/2], [1, 0], [0, -np.sqrt(2)/2]])
    distance = np.round(BinetCauchyDistance().compute_distance(GrassmannPoint(xi), GrassmannPoint(xj)), 6)
    assert distance == 0.996838


def test_asimov_distance():
    xi = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/4], [np.sqrt(2)/2, -np.sqrt(2)/4], [0, -np.sqrt(3)/2]])
    xj = np.array([[0, np.sqrt(2)/2], [1, 0], [0, -np.sqrt(2)/2]])
    distance = np.round(AsimovDistance().compute_distance(GrassmannPoint(xi), GrassmannPoint(xj)), 6)
    assert distance == 1.491253


def test_martin_distance():
    xi = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/4], [np.sqrt(2)/2, -np.sqrt(2)/4], [0, -np.sqrt(3)/2]])
    xj = np.array([[0, np.sqrt(2)/2], [1, 0], [0, -np.sqrt(2)/2]])
    distance = np.round(MartinDistance().compute_distance(GrassmannPoint(xi), GrassmannPoint(xj)), 6)
    assert distance == 2.25056


def test_spectral_distance():
    xi = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/4], [np.sqrt(2)/2, -np.sqrt(2)/4], [0, -np.sqrt(3)/2]])
    xj = np.array([[0, np.sqrt(2)/2], [1, 0], [0, -np.sqrt(2)/2]])
    distance = np.round(SpectralDistance().compute_distance(GrassmannPoint(xi), GrassmannPoint(xj)), 6)
    assert distance == 1.356865


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

    manifold_projection = SvdProjection(matrices, p="max")

    distance_metric = GeodesicDistance()
    value = distance_metric.compute_distance(manifold_projection.psi[0], manifold_projection.psi[1])

    assert value == 1.6024416339920522

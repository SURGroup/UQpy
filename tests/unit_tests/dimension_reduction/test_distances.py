from UQpy.utilities.GrassmannPoint import GrassmannPoint
from UQpy.dimension_reduction.grassmann_manifold.projections.SVDProjection import SVDProjection
from UQpy.utilities.distances.euclidean_distances import L2Distance
from UQpy.utilities.distances.grassmannian_distances import AsimovDistance, BinetCauchyDistance, FubiniStudyDistance, \
    GeodesicDistance, ProcrustesDistance, ProjectionDistance, SpectralDistance
from UQpy.utilities.distances import MartinDistance
from UQpy.utilities.DistanceMetric import DistanceMetric
import numpy as np
import sys


def test_euclidean_distance_2points():
    x = np.array([[2.0, 3.1], [4.0, 1.25]])
    d = L2Distance()
    distance = np.round(d.compute_distance(xi=np.array([2.0, 3.1]), xj=np.array([4.0, 1.25])), 3)

    assert distance == 2.724


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


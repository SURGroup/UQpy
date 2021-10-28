from UQpy.dimension_reduction.distances.euclidean.Euclidean import Euclidean
from UQpy.dimension_reduction.distances.grassmanian.Asimov import Asimov
from UQpy.dimension_reduction.distances.grassmanian.FubiniStudy import FubiniStudy
from UQpy.dimension_reduction.distances.grassmanian.Grassmann import Grassmann
from UQpy.dimension_reduction.distances.grassmanian.Martin import Martin
from UQpy.dimension_reduction.distances.grassmanian.Projection import Projection
from UQpy.dimension_reduction.distances.grassmanian.Spectral import Spectral
from UQpy.dimension_reduction.distances.grassmanian.BinetCauchy import BinetCauchy
from UQpy.dimension_reduction.distances.grassmanian.Procrustes import Procrustes
from UQpy.utilities.DistanceMetric import DistanceMetric
import numpy as np


def test_euclidean_distance_2points():
    x = np.array([[2.0, 3.1], [4.0, 1.25]])
    d = Euclidean(metric=DistanceMetric.EUCLIDEAN)
    distance = np.matrix.round(d.compute_distance(x), 3)

    assert distance == 2.724


def test_euclidean_distance_3points():
    x = np.array([[2.0, 3.1], [2.0, 2.1], [4.0, 1.25]])
    d = Euclidean(metric=DistanceMetric.EUCLIDEAN)
    distance = np.matrix.round(d.compute_distance(x), 3)

    assert np.allclose(distance, np.array([1, 2.724, 2.173]))


def test_grassmann_distance():
    xi = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/4], [np.sqrt(2)/2, -np.sqrt(2)/4], [0, -np.sqrt(3)/2]])
    xj = np.array([[0, np.sqrt(2)/2], [1, 0], [0, -np.sqrt(2)/2]])
    distance = np.round(Grassmann().compute_distance(xi, xj), 6)
    assert distance == 1.491253


def test_fs_distance():
    xi = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/4], [np.sqrt(2)/2, -np.sqrt(2)/4], [0, -np.sqrt(3)/2]])
    xj = np.array([[0, np.sqrt(2)/2], [1, 0], [0, -np.sqrt(2)/2]])
    distance = np.round(FubiniStudy().compute_distance(xi, xj), 6)
    assert distance == 1.491253


def test_procrustes_distance():
    xi = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/4], [np.sqrt(2)/2, -np.sqrt(2)/4], [0, -np.sqrt(3)/2]])
    xj = np.array([[0, np.sqrt(2)/2], [1, 0], [0, -np.sqrt(2)/2]])
    distance = np.round(Procrustes().compute_distance(xi, xj), 6)
    assert distance == 1.356865


def test_projection_distance():
    xi = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/4], [np.sqrt(2)/2, -np.sqrt(2)/4], [0, -np.sqrt(3)/2]])
    xj = np.array([[0, np.sqrt(2)/2], [1, 0], [0, -np.sqrt(2)/2]])
    distance = np.round(Projection().compute_distance(xi, xj), 6)
    assert distance == 0.996838


def test_binet_cauchy_distance():
    xi = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/4], [np.sqrt(2)/2, -np.sqrt(2)/4], [0, -np.sqrt(3)/2]])
    xj = np.array([[0, np.sqrt(2)/2], [1, 0], [0, -np.sqrt(2)/2]])
    distance = np.round(BinetCauchy().compute_distance(xi, xj), 6)
    assert distance == 0.996838


def test_asimov_distance():
    xi = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/4], [np.sqrt(2)/2, -np.sqrt(2)/4], [0, -np.sqrt(3)/2]])
    xj = np.array([[0, np.sqrt(2)/2], [1, 0], [0, -np.sqrt(2)/2]])
    distance = np.round(Asimov().compute_distance(xi, xj), 6)
    assert distance == 1.491253


def test_martin_distance():
    xi = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/4], [np.sqrt(2)/2, -np.sqrt(2)/4], [0, -np.sqrt(3)/2]])
    xj = np.array([[0, np.sqrt(2)/2], [1, 0], [0, -np.sqrt(2)/2]])
    distance = np.round(Martin().compute_distance(xi, xj), 6)
    assert distance == 2.25056


def test_spectral_distance():
    xi = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/4], [np.sqrt(2)/2, -np.sqrt(2)/4], [0, -np.sqrt(3)/2]])
    xj = np.array([[0, np.sqrt(2)/2], [1, 0], [0, -np.sqrt(2)/2]])
    distance = np.round(Spectral().compute_distance(xi, xj), 6)
    assert distance == 1.356865

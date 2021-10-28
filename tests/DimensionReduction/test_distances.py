from UQpy import DistanceMetric
from UQpy.dimension_reduction.distances import Euclidean
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

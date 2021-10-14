import numpy as np
from beartype import beartype
from scipy.spatial.distance import pdist
from UQpy.utilities import DistanceMetric


class Euclidean:
    def __init__(self, metric: DistanceMetric):
        metric_str = str(metric.name).lower()
        self.distance_function = lambda x: pdist(x, metric=metric_str)

    def compute_distance(self, point1, point2) -> float:
        # d = np.linalg.norm(point1 - point2)
        d = self.distance_function([point1, point2])
        return d

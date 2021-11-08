from typing import Union
import numpy as np
from scipy.spatial.distance import pdist
from UQpy.utilities.DistanceMetric import DistanceMetric


class EuclideanDistance:
    def __init__(self, metric: DistanceMetric):
        """
        A class that calculates the Euclidean distance between points.
        :param metric: Enumeration of type DistanceMetric that defines
        the type of distance to be used.

        """
        metric_str = str(metric.name).lower()
        self.distance_function = lambda x: pdist(x, metric=metric_str)

    def compute_distance(self, points: np.array) -> Union[float, np.ndarray]:
        """

        :param numpy.array points: Array holding the coordinates of the points
        :return float or numpy.array: Euclidean Distance
        :rtype float or numpy.array
        """
        d = self.distance_function(points)
        return d

from typing import Union
from scipy.spatial.distance import pdist
from UQpy.utilities import NumpyFloatArray
from UQpy.utilities.DistanceMetric import DistanceMetric


class EuclideanDistance:
    def __init__(self, metric: DistanceMetric):
        """
        A class that calculates the Euclidean distance between points.

        :param metric: Enumeration of type DistanceMetric that defines the type of distance to be used.

        """
        metric_str = str(metric.name).lower()
        self.distance_function = lambda x: pdist(x, metric=metric_str)

    def compute_distance(self, points: NumpyFloatArray) -> Union[float, NumpyFloatArray]:
        """

        :param points: Array holding the coordinates of the points

        """
        return pdist(points, "euclidean")

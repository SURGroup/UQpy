from typing import Union

from UQpy.utilities.ValidationTypes import NumpyFloatArray, Numpy2DFloatArray
from UQpy.utilities.distances.baseclass.EuclideanDistance import EuclideanDistance
from scipy.spatial.distance import pdist


class MinkowskiDistance(EuclideanDistance):
    def __init__(self, p: float = 2):
        """
        :param p: Order of the norm.
        """
        self.p = p

    def compute_distance(self, xi: NumpyFloatArray, xj: NumpyFloatArray) -> float:
        """
        Given two points, this method calculates the Minkowski distance.

        :param xi: First point.
        :param xj: Second point.
        :return: A float representing the distance between the points.
        """
        return pdist([xi, xj], "minkowski", p=self.p)[0]

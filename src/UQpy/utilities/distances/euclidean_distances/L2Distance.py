from typing import Union
from scipy.spatial.distance import pdist
from UQpy.utilities import NumpyFloatArray
from UQpy.utilities.distances.baseclass.EuclideanDistance import EuclideanDistance


class L2Distance(EuclideanDistance):

    def compute_distance(self, xi: NumpyFloatArray, xj: NumpyFloatArray) -> float:
        """
        Given two points, this method calculates the L2 distance.

        :param xi: First point.
        :param xj: Second point.
        :return: A float representing the distance between the points.
        """

        return pdist([xi, xj], "euclidean")[0]

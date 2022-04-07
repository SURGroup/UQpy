from typing import Union

from UQpy.utilities.ValidationTypes import NumpyFloatArray, Numpy2DFloatArray
from UQpy.utilities.distances.baseclass.EuclideanDistance import EuclideanDistance
from scipy.spatial.distance import pdist


class MinkowskiDistance(EuclideanDistance):
    def __init__(self, p=2):
        self.p = p

    def compute_distance(self, xi: NumpyFloatArray, xj: NumpyFloatArray) -> float:
        return pdist([xi, xj], "minkowski", p=self.p)[0]

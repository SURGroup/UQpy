from typing import Union

from UQpy.utilities.ValidationTypes import NumpyFloatArray
from UQpy.utilities.distances.baseclass.EuclideanDistance import EuclideanDistance
from scipy.spatial.distance import pdist


class MinkowskiDistance(EuclideanDistance):
    def __init__(self, p=2):
        self.p = p

    def compute_distance(self, points: NumpyFloatArray) -> Union[float, NumpyFloatArray]:
        return pdist(points, "minkowski", p=self.p)

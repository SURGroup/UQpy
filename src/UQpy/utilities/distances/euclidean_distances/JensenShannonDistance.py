from typing import Union

from UQpy.utilities.ValidationTypes import NumpyFloatArray, Numpy2DFloatArray
from UQpy.utilities.distances.baseclass.EuclideanDistance import EuclideanDistance
from scipy.spatial.distance import pdist


class JensenShannonDistance(EuclideanDistance):

    def compute_distance(self, xi: NumpyFloatArray, xj: NumpyFloatArray) -> float:
        return pdist([xi, xj], "jensenshannon")[0]

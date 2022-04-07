from typing import Union
from scipy.spatial.distance import pdist
from UQpy.utilities import NumpyFloatArray


class EuclideanDistance:

    def compute_distance(self, xi: NumpyFloatArray, xj: NumpyFloatArray) -> float:
        return pdist([xi, xj], "euclidean")[0]

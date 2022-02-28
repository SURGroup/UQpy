import itertools
from abc import ABC

import numpy as np
from beartype import beartype

from UQpy.utilities.ValidationTypes import NumpyFloatArray
from UQpy.utilities.distances.baseclass.Distance import Distance


class EuclideanDistance(Distance, ABC):

    def calculate_distance_matrix(self, points):
        pass

    @beartype
    def calculate_distance_matrix(self, points: list[NumpyFloatArray]):
        nargs = len(points)

        # Define the pairs of points to compute the grassmann_manifold distance.
        indices = range(nargs)
        pairs = list(itertools.combinations(indices, 2))

        # Compute the pairwise distances.
        distance_list = []
        for id_pair in range(np.shape(pairs)[0]):
            ii = pairs[id_pair][0]  # Point i
            jj = pairs[id_pair][1]  # Point j

            x0 = points[ii]
            x1 = points[jj]

            # Call the functions where the distance metric is implemented.
            distance_value = self.compute_distance(x0, x1)

            distance_list.append(distance_value)

        return distance_list

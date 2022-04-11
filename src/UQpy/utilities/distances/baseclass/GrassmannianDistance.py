import itertools
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from beartype import beartype

from UQpy.utilities.GrassmannPoint import GrassmannPoint
from UQpy.utilities.ValidationTypes import Numpy2DFloatArrayOrthonormal
from UQpy.utilities.distances.baseclass.Distance import Distance


class GrassmannianDistance(Distance, ABC):
    @staticmethod
    @beartype
    def check_rows(xi, xj):
        if xi.data.shape[0] != xj.data.shape[0]:
            raise ValueError("UQpy: Incompatible dimensions. The matrices must have the same number of rows.")

    @beartype
    def calculate_distance_matrix(self,
                                  points: Union[list[Numpy2DFloatArrayOrthonormal],  list[GrassmannPoint]],
                                  p_dim: Union[list, np.ndarray]):
        """
        Given a list of points that belong on a Grassmann Manifold, assemble the distance matrix between all points.

        :param points: List of points belonging on the Grassmann Manifold. Either a list of :class:`.GrassmannPoint` or
         a list of orthonormal :class:`.ndarray`.
        :param p_dim: Number of independent p-planes of each Grassmann point.
        """
        nargs = len(points)

        # Define the pairs of points to compute the grassmann_manifold distance.
        indices = range(nargs)
        pairs = list(itertools.combinations(indices, 2))

        # Compute the pairwise distances.
        distance_list = []
        for id_pair in range(np.shape(pairs)[0]):
            ii = pairs[id_pair][0]  # Point i
            jj = pairs[id_pair][1]  # Point j

            p0 = int(p_dim[ii])
            p1 = int(p_dim[jj])

            x0 = GrassmannPoint(np.asarray(points[ii].data)[:, :p0])
            x1 = GrassmannPoint(np.asarray(points[jj].data)[:, :p1])

            # Call the functions where the distance metric is implemented.
            distance_value = self.compute_distance(x0, x1)

            distance_list.append(distance_value)

        self.distance_matrix = distance_list

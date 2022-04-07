from abc import abstractmethod, ABC
from typing import Union

import numpy as np



class Distance(ABC):
    """
    This class serves as a baseclass under which the distance implementations for all spaces exist.
    """

    def calculate_distance_matrix(self, points):
        """
        Using the distance-specific :py:meth:`.compute_distance` method, this function assembles the distance matrix.

        :param points: Set of data points. Depending on the type of kernel these should be either :class:`numpy.ndarray`
            or of type :class:`.GrassmannPoint`.
        """
        pass

    @abstractmethod
    def compute_distance(self, xi, xj) -> float:
        """
        Given a two points, this method calculates their distance. Each concrete distance implementation
        must override this method and provide its own implementation.

        :param xi: First point.
        :param xj: Second point.
        :return: A float representing the distance between the points.
        """
        pass

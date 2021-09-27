from abc import ABC, abstractmethod
import numpy as np


class RiemannianDistance(ABC):

    @abstractmethod
    def compute_distance(self, point1, point2):
        pass

    @staticmethod
    def check_points(point1, point2):
        if not isinstance(point1, list) and not isinstance(point1, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            point1 = np.array(point1)

        if not isinstance(point2, list) and not isinstance(point2, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            point2 = np.array(point2)

        return point1, point2

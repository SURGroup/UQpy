from abc import ABC, abstractmethod
import numpy as np


class RiemannianDistance(ABC):
    @abstractmethod
    def compute_distance(self, xi, xj):
        pass

    @staticmethod
    def check_points(xi, xj):
        if isinstance(xi, np.ndarray):
            if len(xj.shape) != 2:
                raise TypeError("UQpy: Point on the Grassmann must be given as a 2D numpy.ndarray.")
            else:
                if not np.allclose(xi.T @ xi, np.eye(xi.shape[1])):
                    raise TypeError("UQpy: Point Xi must lie on a Grassmann manifold, i.e., X'X=I.")
        elif isinstance(xi, list):
            raise TypeError("UQpy: Point on the Grassmann must be given as a 2D numpy.ndarray.")

        if isinstance(xj, np.ndarray):
            if len(xj.shape) != 2:
                raise TypeError("UQpy: Point on the Grassmann must be given as a 2D numpy.ndarray.")
            else:
                if not np.allclose(xj.T @ xj, np.eye(xj.shape[1])):
                    raise TypeError("UQpy: Point Xj must lie on a Grassmann manifold, i.e., X'X=I.")
        elif isinstance(xi, list):
            raise TypeError("UQpy: Point on the Grassmann must be given as a 2D numpy.ndarray.")

        if xi.shape[0] != xj.shape[0]:
            raise ValueError("UQpy: Incompatible dimensions. The matrices must have the same number of rows.")

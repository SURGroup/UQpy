import numpy as np


class Euclidean:

    def compute_distance(self, point1, point2) -> float:
        d = np.linalg.norm(point1 - point2)
        return d

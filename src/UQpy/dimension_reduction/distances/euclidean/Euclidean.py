import numpy as np


class Euclidean:

    def compute_distance(self, point1, point2):
        """
               euclidean distance.

               **Input:**

               * **x0** (`list` or `ndarray`)
                   Point.

               * **x1** (`list` or `ndarray`)
                   Point.

               **Output/Returns:**

               * **d** (`float`)
                   Distance between x0 and x1.

               """

        d = np.linalg.norm(point1 - point2)

        return d

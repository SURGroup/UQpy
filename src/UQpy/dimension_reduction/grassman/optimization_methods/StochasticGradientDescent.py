from UQpy import PositiveInteger
import numpy as np

from UQpy.dimension_reduction.distances.grassmanian.baseclass.RiemannianDistance import RiemannianDistance
from UQpy.dimension_reduction.grassman.KarcherMean import KarcherMean


class StochasticGradientDescent:

    def __init__(self, acceleration: bool = False,
                 error_tolerance: float = 1e-3,
                 max_iterations:PositiveInteger = 1000):
        self.max_iterations = max_iterations
        self.error_tolerance = error_tolerance
        self.acceleration = acceleration

    def optimize(self, data_points, distance):

        n_mat = len(data_points)

        rnk = []
        for i in range(n_mat):
            rnk.append(min(np.shape(data_points[i])))

        max_rank = max(rnk)

        fmean = []
        for i in range(n_mat):
            fmean.append(KarcherMean.frechet_variance(data_points[i]))

        index_0 = fmean.index(min(fmean))

        mean_element = data_points[index_0].tolist()
        counter_iteration = 0
        _gamma = []
        k = 1
        from UQpy.dimension_reduction.grassman.Grassman import Grassmann
        while counter_iteration < self.max_iterations:

            indices = np.arange(n_mat)
            np.random.shuffle(indices)

            melem = mean_element
            for i in range(len(indices)):
                alpha = 0.5 / k
                idx = indices[i]
                _gamma = Grassmann.log_map(points_grassmann=[data_points[idx]], ref=np.asarray(mean_element))

                step = 2 * alpha * _gamma[0]

                X = Grassmann.exp_map(points_tangent=[step], ref=np.asarray(mean_element))

                _gamma = []
                mean_element = X[0]

                k += 1

            test_1 = np.linalg.norm(mean_element - melem, 'fro')
            if test_1 < self.error_tolerance:
                break

            counter_iteration += 1

        return mean_element
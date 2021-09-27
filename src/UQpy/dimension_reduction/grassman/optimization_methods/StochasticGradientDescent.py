from UQpy import PositiveInteger
from UQpy.dimension_reduction_v4.kernel_based.distances.baseclass.RiemannianDistance import RiemannianDistance
import numpy as np


class StochasticGradientDescent:

    def __init__(self, distance_function: RiemannianDistance,
                 acceleration: bool = False,
                 error_tolerance: float = 1e-3,
                 max_iterations:PositiveInteger = 1000):
        self.max_iterations = max_iterations
        self.error_tolerance = error_tolerance
        self.acceleration = acceleration
        self.distance_function = distance_function

    def optimize(self, data_points):

        data_points = self.X

        if 'tol' in kwargs.keys():
            tol = kwargs['tol']
        else:
            tol = 1e-3

        if 'maxiter' in kwargs.keys():
            maxiter = kwargs['maxiter']
        else:
            maxiter = 1000

        n_mat = len(data_points)

        rnk = []
        for i in range(n_mat):
            rnk.append(min(np.shape(data_points[i])))

        max_rank = max(rnk)

        fmean = []
        for i in range(n_mat):
            fmean.append(self.frechet_variance(data_points[i]))

        index_0 = fmean.index(min(fmean))

        mean_element = data_points[index_0].tolist()
        itera = 0
        _gamma = []
        k = 1
        while itera < maxiter:

            indices = np.arange(n_mat)
            np.random.shuffle(indices)

            melem = mean_element
            for i in range(len(indices)):
                alpha = 0.5 / k
                idx = indices[i]
                _gamma = self.log_map(points_grassmann=[data_points[idx]], ref=np.asarray(mean_element))

                step = 2 * alpha * _gamma[0]

                X = self.exp_map(points_tangent=[step], ref=np.asarray(mean_element))

                _gamma = []
                mean_element = X[0]

                k += 1

            test_1 = np.linalg.norm(mean_element - melem, 'fro')
            if test_1 < tol:
                break

            itera += 1

        return mean_element
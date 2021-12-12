from UQpy.optimization.baseclass.OptimizationMethod import OptimizationMethod
from UQpy.utilities.ValidationTypes import PositiveInteger
import numpy as np


class StochasticGradientDescent(OptimizationMethod):
    def __init__(
        self,
        acceleration: bool = False,
        error_tolerance: float = 1e-3,
        max_iterations: PositiveInteger = 1000,
    ):
        """

        :param acceleration:
        :param error_tolerance:
        :param max_iterations:
        """
        self.max_iterations = max_iterations
        self.error_tolerance = error_tolerance
        self.acceleration = acceleration

    def optimize(self, data_points, distance):
        from UQpy.dimension_reduction.grassmann_manifold.Grassmann import Grassmann
        n_mat = len(data_points)

        rnk = []
        for i in range(n_mat):
            rnk.append(min(np.shape(data_points[i])))

        max_rank = max(rnk)

        fmean = []
        for i in range(n_mat):
            fmean.append(Grassmann.frechet_variance(data_points[i]))

        index_0 = fmean.index(min(fmean))

        mean_element = data_points[index_0].tolist()
        counter_iteration = 0
        _gamma = []
        k = 1


        while counter_iteration < self.max_iterations:

            indices = np.arange(n_mat)
            np.random.shuffle(indices)

            melem = mean_element
            for i in range(len(indices)):
                alpha = 0.5 / k
                idx = indices[i]
                _gamma = Grassmann.log_map(
                    manifold_points=[data_points[idx]],
                    reference_point=np.asarray(mean_element),
                )

                step = 2 * alpha * _gamma[0]

                X = Grassmann.exp_map(
                    tangent_points=[step], reference_point=np.asarray(mean_element)
                )

                _gamma = []
                mean_element = X[0]

                k += 1

            test_1 = np.linalg.norm(mean_element - melem, "fro")
            if test_1 < self.error_tolerance:
                break

            counter_iteration += 1

        return mean_element

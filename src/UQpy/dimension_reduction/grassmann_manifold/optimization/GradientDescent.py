import copy
from UQpy.utilities.ValidationTypes import PositiveInteger
import numpy as np


class GradientDescent:
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
        # Number of points.
        points_number = len(data_points)
        alpha = 0.5
        rank = []
        for i in range(points_number):
            rank.append(min(np.shape(data_points[i])))

        from UQpy.dimension_reduction.grassmann_manifold.Grassmann import Grassmann

        max_rank = max(rank)
        fmean = []
        for i in range(points_number):
            fmean.append(
                Grassmann.frechet_variance(reference_point=data_points[i], manifold_points=data_points,
                                           distance=distance)
            )

        index_0 = fmean.index(min(fmean))
        mean_element = data_points[index_0].tolist()

        avg_gamma = np.zeros([np.shape(data_points[0])[0], np.shape(data_points[0])[1]])

        counter_iteration = 0

        l = 0
        avg = []
        _gamma = []
        if self.acceleration:
            _gamma = Grassmann.log_map(
                manifold_points=data_points, reference_point=np.asarray(mean_element)
            )

            avg_gamma.fill(0)
            for i in range(points_number):
                avg_gamma += _gamma[i] / points_number
            avg.append(avg_gamma)

        # Main loop
        while counter_iteration <= self.max_iterations:
            _gamma = Grassmann.log_map(
                manifold_points=data_points, reference_point=np.asarray(mean_element)
            )
            avg_gamma.fill(0)

            for i in range(points_number):
                avg_gamma += _gamma[i] / points_number

            test_0 = np.linalg.norm(avg_gamma, "fro")
            if test_0 < self.error_tolerance and counter_iteration == 0:
                break

            # Nesterov: Accelerated Gradient Descent
            if self.acceleration:
                avg.append(avg_gamma)
                l0 = l
                l1 = 0.5 * (1 + np.sqrt(1 + 4 * l * l))
                ls = (1 - l0) / l1
                step = (1 - ls) * avg[counter_iteration + 1] + ls * avg[
                    counter_iteration
                ]
                l = copy.copy(l1)
            else:
                step = alpha * avg_gamma

            x = Grassmann.exp_map(
                tangent_points=[step], reference_point=np.asarray(mean_element)
            )

            test_1 = np.linalg.norm(x[0] - mean_element, "fro")

            if test_1 < self.error_tolerance:
                break

            mean_element = []
            mean_element = x[0]

            counter_iteration += 1

        # return the Karcher mean.
        return mean_element

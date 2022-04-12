from beartype import beartype

from UQpy.sampling.adaptive_kriging_functions.baseclass.LearningFunction import (
    LearningFunction,
)
import numpy as np


class WeightedUFunction(LearningFunction):

    @beartype
    def __init__(self, weighted_u_stop: int):
        """
        Probability Weighted U-function for reliability analysis. See :cite:`AKMCS3` for a detailed explanation.

        :param weighted_u_stop: Stopping parameter required for the WeightedU learning function
        """
        self.weighted_u_stop = weighted_u_stop

    def evaluate_function(self, distributions, n_add, surrogate, population, qoi=None, samples=None):
        g, sig = surrogate.predict(population, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([population.shape[0], 1])
        sig = sig.reshape([population.shape[0], 1])

        u = abs(g) / sig
        p1 = np.ones([population.shape[0], population.shape[1]])
        p2 = np.ones([samples.shape[0], population.shape[1]])

        for j in range(samples.shape[1]):
            p1[:, j] = distributions[j].pdf(np.atleast_2d(population[:, j]).T)
            p2[:, j] = distributions[j].pdf(np.atleast_2d(samples[:, j]).T)

        p1 = p1.prod(1).reshape(u.size, 1)
        max_p = max(p2.prod(1))
        u_ = u * ((max_p - p1) / max_p)
        rows = u_[:, 0].argsort()[:n_add]

        stopping_criteria_indicator = min(u[:, 0]) >= self.weighted_u_stop
        new_samples = population[rows, :]
        learning_function_values = u_[rows, :]
        return new_samples, learning_function_values, stopping_criteria_indicator

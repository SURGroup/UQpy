from beartype import beartype

from UQpy.sampling.adaptive_kriging_functions.baseclass.LearningFunction import (
    LearningFunction,
)


class UFunction(LearningFunction):

    @beartype
    def __init__(self, u_stop: int = 2):
        """
        U-function for reliability analysis. See :cite:`AKMCS1` for a detailed explanation.

        :param u_stop: U-Function stopping parameter
        """
        self.u_stop = u_stop

    def evaluate_function(self, distributions, n_add, surrogate, population, qoi=None, samples=None):

        g, sig = surrogate.predict(population, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([population.shape[0], 1])
        sig = sig.reshape([population.shape[0], 1])

        u = abs(g) / sig
        rows = u[:, 0].argsort()[:n_add]

        stopping_criteria_indicator = min(u[:, 0]) >= self.u_stop
        new_samples = population[rows, :]
        learning_function_values = u[rows, 0]
        return new_samples, learning_function_values, stopping_criteria_indicator

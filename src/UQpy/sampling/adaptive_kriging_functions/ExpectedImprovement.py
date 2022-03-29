from typing import Union
from beartype import beartype
from UQpy.sampling.adaptive_kriging_functions.baseclass.LearningFunction import (
    LearningFunction,
)
import scipy.stats as stats
import numpy as np


class ExpectedImprovement(LearningFunction):

    @beartype
    def __init__(self, eif_stop: Union[float, int] = 0.01):
        """
        Expected Improvement Function (EIF) for Efficient Global Optimization (EFO). See :cite:`AKMCS2` for a detailed
        explanation.

        :param eif_stop: Stopping threshold
        """
        self.eif_stop = eif_stop

    def evaluate_function(self, distributions, n_add, surrogate, population, qoi=None, samples=None):
        g, sig = surrogate.predict(population, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([population.shape[0], 1])
        sig = sig.reshape([population.shape[0], 1])

        fm = min(qoi)
        eif = (fm - g) * stats.norm.cdf((fm - g) / sig) + sig * stats.norm.pdf((fm - g) / sig)
        rows = eif[:, 0].argsort()[(np.size(g) - n_add) :]

        stopping_criteria_indicator = max(eif[:, 0]) / abs(fm) <= self.eif_stop
        new_samples = population[rows, :]
        learning_function_values = eif[rows, :]
        return new_samples, learning_function_values, stopping_criteria_indicator

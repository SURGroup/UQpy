from typing import Union

from beartype import beartype

from UQpy.sampling.adaptive_kriging_functions.baseclass.LearningFunction import (
    LearningFunction,
)
import scipy.stats as stats


class ExpectedFeasibility(LearningFunction):

    @beartype
    def __init__(
        self,
        eff_a: Union[float, int] = 0,
        eff_epsilon: Union[float, int] = 2,
        eff_stop: Union[float, int] = 0.001,
    ):
        """
        Expected Feasibility Function (EFF) for reliability analysis, see :cite:`AKMCS4` for a detailed explanation.

        :param eff_a: Reliability threshold.
        :param eff_epsilon: EGRA method epsilon
        :param eff_stop: Stopping threshold
        """
        self.eff_a = eff_a
        self.eff_epsilon = eff_epsilon
        self.eff_stop = eff_stop

    def evaluate_function(self, distributions, n_add, surrogate, population, qoi=None, samples=None):

        g, sig = surrogate.predict(population, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([population.shape[0], 1])
        sig = sig.reshape([population.shape[0], 1])
        # reliability threshold: a_ = 0
        # EGRA method: epsilon = 2*sigma(x)
        a_, ep = self.eff_a, self.eff_epsilon * sig
        t1 = (a_ - g) / sig
        t2 = (a_ - ep - g) / sig
        t3 = (a_ + ep - g) / sig
        eff = (g - a_) * (
            2 * stats.norm.cdf(t1) - stats.norm.cdf(t2) - stats.norm.cdf(t3)
        )
        eff += -sig * (2 * stats.norm.pdf(t1) - stats.norm.pdf(t2) - stats.norm.pdf(t3))
        eff += ep * (stats.norm.cdf(t3) - stats.norm.cdf(t2))
        rows = eff[:, 0].argsort()[-n_add:]

        stopping_criteria_indicator = max(eff[:, 0]) <= self.eff_stop
        new_samples = population[rows, :]
        learning_function_values = eff[rows, :]
        return new_samples, learning_function_values, stopping_criteria_indicator

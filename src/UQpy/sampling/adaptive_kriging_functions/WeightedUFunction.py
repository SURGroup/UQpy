from beartype import beartype

from UQpy.sampling.adaptive_kriging_functions.baseclass.LearningFunction import (
    LearningFunction,
)
import numpy as np


class WeightedUFunction(LearningFunction):
    """
            Probability Weighted U-function for reliability analysis. See [5]_ for a detailed explanation.


            **Inputs:**

            * **surr** (`class` object):
                A kriging surrogate model, this object must have a ``predict`` method as defined in `krig_object`
                parameter.

            * **pop** (`ndarray`):
                An array of samples defining the learning set at which points the weighted U-function is evaluated

            * **n_add** (`int`):
                Number of samples to be added per iteration.

                Default: 1.

            * **parameters** (`dictionary`)
                Dictionary containing all necessary parameters and the stopping criterion for the learning function.
                Here this includes the parameter `u_stop`.

            * **samples** (`ndarray`):
                The initial samples at which to evaluate the model.

            * **qoi** (`list`):
                A list, which contaains the model evaluations.

            * **dist_object** ((list of) ``Distribution`` object(s)):
                List of ``Distribution`` objects corresponding to each random variable.

            **Output/Returns:**

            * **new_samples** (`ndarray`):
                Samples selected for model evaluation.

            * **w_lf** (`ndarray`)
                Weighted U learning function evaluated at the new sample points.

            * **indicator** (`boolean`):
                Indicator for stopping criteria.

                `indicator = True` specifies that the stopping criterion has been met and the AKMCS.run method stops.

            """

    @beartype
    def __init__(self, weighted_u_stop: int):
        self.weighted_u_stop = weighted_u_stop

    def evaluate_function(
        self, distributions, n_add, surrogate, population, qoi=None, samples=None
    ):
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

        stopping_criteria_indicator = False
        if min(u[:, 0]) >= self.weighted_u_stop:
            stopping_criteria_indicator = True

        new_samples = population[rows, :]
        learning_function_values = u_[rows, :]
        return new_samples, learning_function_values, stopping_criteria_indicator

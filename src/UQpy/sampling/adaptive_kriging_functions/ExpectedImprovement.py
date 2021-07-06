from UQpy.sampling.adaptive_kriging_functions.baseclass.LearningFunction import LearningFunction
import scipy.stats as stats
import numpy as np


class ExpectedImprovement(LearningFunction):
    """
            Expected Improvement Function (EIF) for Efficient Global Optimization (EFO). See [4]_ for a detailed
            explanation.


            **Inputs:**

            * **surr** (`class` object):
                A kriging surrogate model, this object must have a ``predict`` method as defined in `krig_object`
                parameter.

            * **pop** (`ndarray`):
                An array of samples defining the learning set at which points the EIF is evaluated

            * **n_add** (`int`):
                Number of samples to be added per iteration.

                Default: 1.

            * **parameters** (`dictionary`)
                Dictionary containing all necessary parameters and the stopping criterion for the learning function.
                Here this includes the parameter `eif_stop`.

            * **samples** (`ndarray`):
                The initial samples at which to evaluate the model.

            * **qoi** (`list`):
                A list, which contaains the model evaluations.

            * **dist_object** ((list of) ``Distribution`` object(s)):
                List of ``Distribution`` objects corresponding to each random variable.


            **Output/Returns:**

            * **new_samples** (`ndarray`):
                Samples selected for model evaluation.

            * **indicator** (`boolean`):
                Indicator for stopping criteria.

                `indicator = True` specifies that the stopping criterion has been met and the AKMCS.run method stops.

            * **eif_lf** (`ndarray`)
                EIF learning function evaluated at the new sample points.
            """
    def __init__(self, eif_stop=0.01):
        self.eif_stop = eif_stop

    def evaluate_function(self, distributions, n_add, surrogate, population, qoi=None, samples=None):
        g, sig = surrogate.predict(population, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([population.shape[0], 1])
        sig = sig.reshape([population.shape[0], 1])

        fm = min(qoi)
        eif = (fm - g) * stats.norm.cdf((fm - g) / sig) + sig * stats.norm.pdf((fm - g) / sig)
        rows = eif[:, 0].argsort()[(np.size(g) - n_add):]

        stopping_criteria_indicator = False
        if max(eif[:, 0]) / abs(fm) <= self.eif_stop:
            stopping_criteria_indicator = True

        new_samples = population[rows, :]
        learning_function_values = eif[rows, :]
        return new_samples, learning_function_values, stopping_criteria_indicator

from beartype import beartype

from UQpy.sampling.adaptive_kriging_functions.baseclass.LearningFunction import (
    LearningFunction,
)


class UFunction(LearningFunction):
    """
            U-function for reliability analysis. See [3] for a detailed explanation.


            **Inputs:**

            * **surr** (`class` object):
                A kriging surrogate model, this object must have a ``predict`` method as defined in `krig_object`
                parameter.

            * **pop** (`ndarray`):
                An array of samples defining the learning set at which points the U-function is evaluated

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

            * **indicator** (`boolean`):
                Indicator for stopping criteria.

                `indicator = True` specifies that the stopping criterion has been met and the AKMCS.run method stops.

            * **u_lf** (`ndarray`)
                U learning function evaluated at the new sample points.

            """

    @beartype
    def __init__(self, u_stop: int = 2):
        self.u_stop = u_stop

    def evaluate_function(
        self, distributions, n_add, surrogate, population, qoi=None, samples=None
    ):

        g, sig = surrogate.predict(population, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([population.shape[0], 1])
        sig = sig.reshape([population.shape[0], 1])

        u = abs(g) / sig
        rows = u[:, 0].argsort()[:n_add]

        stopping_criteria_indicator = False
        if min(u[:, 0]) >= self.u_stop:
            stopping_criteria_indicator = True

        new_samples = population[rows, :]
        learning_function_values = u[rows, 0]
        return new_samples, learning_function_values, stopping_criteria_indicator

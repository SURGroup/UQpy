from UQpy.sampling.adaptive_kriging_functions.baseclass.LearningFunction import LearningFunction
import numpy as np
from sklearn.neighbors import NearestNeighbors


class ExpectedImprovementGlobalFit(LearningFunction):
    """
            Expected Improvement for Global Fit (EIGF) learning function. See [7]_ for a detailed explanation.


            **Inputs:**

            * **surr** (`class` object):
                A kriging surrogate model, this object must have a ``predict`` method as defined in `krig_object`
                parameter.

            * **pop** (`ndarray`):
                An array of samples defining the learning set at which points the EIGF is evaluated

            * **n_add** (`int`):
                Number of samples to be added per iteration.

                Default: 1.

            * **parameters** (`dictionary`)
                Dictionary containing all necessary parameters and the stopping criterion for the learning function. For
                ``EIGF``, this dictionary is empty as no stopping criterion is specified.

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

            * **eigf_lf** (`ndarray`)
                EIGF learning function evaluated at the new sample points.

            """

    def __init__(self, surrogate, pop, samples, qoi, n_add):
        self.surrogate = surrogate
        self.pop = pop
        self.samples = samples
        self.qoi = qoi
        self.n_add = n_add

    def evaluate_function(self):
        g, sig = self.surrogate(self.pop, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([self.pop.shape[0], 1])
        sig = sig.reshape([self.pop.shape[0], 1])

        # Evaluation of the learning function
        # First, find the nearest neighbor in the training set for each point in the population.

        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(np.atleast_2d(self.samples))
        neighbors = knn.kneighbors(np.atleast_2d(self.pop), return_distance=False)

        # noinspection PyTypeChecker
        qoi_array = np.array([self.qoi[x] for x in np.squeeze(neighbors)])

        # Compute the learning function at every point in the population.
        u = np.square(g - qoi_array) + np.square(sig)
        rows = u[:, 0].argsort()[(np.size(g) - self.n_add):]

        stopping_criteria_indicator = False
        new_samples = self.pop[rows, :]
        learning_function_evaluations = u[rows, :]

        return new_samples, learning_function_evaluations, stopping_criteria_indicator

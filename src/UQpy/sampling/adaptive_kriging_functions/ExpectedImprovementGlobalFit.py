from UQpy.sampling.adaptive_kriging_functions.baseclass.LearningFunction import (
    LearningFunction,
)
import numpy as np
from sklearn.neighbors import NearestNeighbors


class ExpectedImprovementGlobalFit(LearningFunction):
    """
    Expected Improvement for Global Fit (EIGF) learning function. See :cite:`AKMCS5` for a detailed explanation.
    """

    def evaluate_function(
        self, distributions, n_add, surrogate, population, qoi=None, samples=None
    ):
        g, sig = surrogate.predict(population, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([population.shape[0], 1])
        sig = sig.reshape([population.shape[0], 1])

        # Evaluation of the learning function
        # First, find the nearest neighbor in the training set for each point in the population.

        knn = NearestNeighbors(n_neighbors=1)
        knn.fit(np.atleast_2d(samples))
        neighbors = knn.kneighbors(np.atleast_2d(population), return_distance=False)

        # noinspection PyTypeChecker
        qoi_array = np.array([qoi[x] for x in np.squeeze(neighbors)])

        # Compute the learning function at every point in the population.
        u = np.square(g - qoi_array) + np.square(sig)
        rows = u[:, 0].argsort()[(np.size(g) - n_add) :]

        stopping_criteria_indicator = False
        new_samples = population[rows, :]
        learning_function_evaluations = u[rows, :]

        return new_samples, learning_function_evaluations, stopping_criteria_indicator

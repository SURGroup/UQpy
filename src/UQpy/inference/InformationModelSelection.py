import logging
from typing import Union

from UQpy.inference.information_criteria import AIC
from UQpy.inference.information_criteria.baseclass.InformationCriterion import InformationCriterion
from beartype import beartype
from UQpy.inference.MLE import MLE
import numpy as np


class InformationModelSelection:

    # Authors: Audrey Olivier, Dimitris Giovanis
    # Last Modified: 12/19 by Audrey Olivier
    @beartype
    def __init__(
            self,
            parameter_estimators: list[MLE],
            criterion: InformationCriterion = AIC(),
            n_optimizations: list[int] = None,
            initial_parameters: list[np.ndarray] = None
    ):
        """
        Perform model selection using information theoretic criteria.

        Supported criteria are :class:`.BIC`, :class:`.AIC` (default), :class:`.AICc`. This class leverages the
        :class:`.MLE` class for maximum likelihood estimation, thus inputs to :class:`.MLE` can also be provided to
        :class:`InformationModelSelection`, as lists of length equal to the number of models.

        :param parameter_estimators: A list containing a maximum-likelihood estimator (:class:`.MLE`) for each one of the
         models to be compared.
        :param criterion: Criterion to be used (:class:`.AIC`, :class:`.BIC`, :class:`.AICc)`. Default is :class:`.AIC`
        :param initial_parameters: Initial guess(es) for optimization, :class:`numpy.ndarray` of shape
         :code:`(nstarts, n_parameters)` or :code:`(n_parameters, )`, where :code:`nstarts` is the number of times the
         optimizer will be called. Alternatively, the user can provide input `n_optimizations` to randomly sample
         initial guess(es). The identified MLE is the one that yields the maximum log likelihood over all calls of the
         optimizer.
        """
        self.candidate_models = [mle.inference_model for mle in parameter_estimators]
        self.models_number = len(parameter_estimators)
        self.criterion: InformationCriterion = criterion
        self.logger = logging.getLogger(__name__)

        self.n_optimizations = n_optimizations
        self.initial_parameters= initial_parameters

        self.parameter_estimators: list = parameter_estimators
        """:class:`.MLE` results for each model (contains e.g. fitted parameters)"""

        # Initialize the outputs
        self.criterion_values: list = [None, ] * self.models_number
        """Value of the criterion for all models."""
        self.penalty_terms: list = [None, ] * self.models_number
        """Value of the penalty term for all models. Data fit term is then criterion_value - penalty_term."""
        self.probabilities: list = [None, ] * self.models_number
        """Value of the model probabilities, computed as
        
        .. math:: P(M_i|d) = \dfrac{\exp(-\Delta_i/2)}{\sum_i \exp(-\Delta_i/2)}
        
        where :math:`\Delta_i = criterion_i - min_i(criterion)`"""

        # Run the model selection procedure
        if (self.n_optimizations is not None) or (self.initial_parameters is not None):
            self.run(self.n_optimizations, self.initial_parameters)

    def run(self, n_optimizations: list[int], initial_parameters: list[np.ndarray]=None):
        """
        Run the model selection procedure, i.e. compute criterion value for all models.

        This function calls the :meth:`run` method of the :class:`.MLE` object for each model to compute the maximum
        log-likelihood, then computes the criterion value and probability for each model. If `data` are given when
        creating the :class:`.MLE` object, this method is called automatically when the object is created.

        :param n_optimizations: Number of iterations that the optimization is run, starting at random initial
         guesses. It is only used if `initial_parameters` is not provided. Default is :math:`1`.
         The random initial guesses are sampled uniformly between :math:`0` and :math:`1`, or uniformly between
         user-defined bounds if an input bounds is provided as a keyword argument to the `optimizer` input parameter.
        :param initial_parameters: Initial guess(es) for optimization, :class:`numpy.ndarray` of shape
         :code:`(nstarts, n_parameters)` or :code:`(n_parameters, )`, where :code:`nstarts` is the number of times the
         optimizer will be called. Alternatively, the user can provide input `n_optimizations` to randomly sample
         initial guess(es). The identified MLE is the one that yields the maximum log likelihood over all calls of the
         optimizer.
        """
        if (n_optimizations is not None and (len(n_optimizations) != len(self.parameter_estimators))) or \
           (initial_parameters is not None and len(initial_parameters) != len(self.parameter_estimators)):
            raise ValueError("The length of n_optimizations and initial_parameters should be equal to the number of "
                             "parameter estimators")
        # Loop over all the models
        for i, parameter_estimator in enumerate(self.parameter_estimators):
            # First evaluate ML estimate for all models, do several iterations if demanded
            parameters = None
            if initial_parameters is not None:
                parameters = initial_parameters[i]

            optimizations = 0
            if n_optimizations is not None:
                optimizations = n_optimizations[i]

            parameter_estimator.run(n_optimizations=optimizations, initial_parameters=parameters)

            # Then minimize the criterion
            self.criterion_values[i], self.penalty_terms[i] = \
                self.criterion.minimize_criterion(data=parameter_estimator.data,
                                                  parameter_estimator=parameter_estimator,
                                                  return_penalty=True)
        # Compute probabilities from criterion values
        self.probabilities = self._compute_probabilities(self.criterion_values)

    def sort_models(self):
        """
        Sort models in descending order of model probability (increasing order of `criterion` value).

        This function sorts - in place - the attribute lists :py:attr:`.candidate_models`, :py:attr:`.ml_estimators`,
        :py:attr:`criterion_values`, :py:attr:`penalty_terms` and :py:attr:`probabilities` so that they are sorted from
        most probable to least probable model. It is a stand-alone function that is provided to help the user to easily
        visualize which model is the best.

        No inputs/outputs.

        """
        sort_idx = list(np.argsort(np.array(self.criterion_values)))

        self.candidate_models = [self.candidate_models[i] for i in sort_idx]
        self.parameter_estimators = [self.parameter_estimators[i] for i in sort_idx]
        self.criterion_values = [self.criterion_values[i] for i in sort_idx]
        self.penalty_terms = [self.penalty_terms[i] for i in sort_idx]
        self.probabilities = [self.probabilities[i] for i in sort_idx]

    @staticmethod
    def _compute_probabilities(criterion_values):
        delta = np.array(criterion_values) - min(criterion_values)
        prob = np.exp(-delta / 2)
        return prob / np.sum(prob)

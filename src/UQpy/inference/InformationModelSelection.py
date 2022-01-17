import logging
from typing import Union

from UQpy.inference.information_criteria import AIC
from UQpy.inference.information_criteria.baseclass.InformationCriterion import InformationCriterion
from UQpy.optimization.MinimizeOptimizer import MinimizeOptimizer
from UQpy.optimization.baseclass.Optimizer import Optimizer
from beartype import beartype
from UQpy.inference.inference_models.baseclass.InferenceModel import InferenceModel
from UQpy.inference.MLE import MLE
from UQpy.utilities.ValidationTypes import RandomStateType, PositiveInteger, NumpyFloatArray
from UQpy.utilities.Utilities import process_random_state
import numpy as np


class InformationModelSelection:

    # Authors: Audrey Olivier, Dimitris Giovanis
    # Last Modified: 12/19 by Audrey Olivier
    @beartype
    def __init__(
            self,
            mle_estimators: list[MLE],
            data: Union[list, np.ndarray] = None,
            criterion: InformationCriterion = AIC(),
    ):
        """
        Perform model selection using information theoretic criteria.

        Supported criteria are :class:`.BIC`, :class:`.AIC` (default), :class:`.AICc`. This class leverages the
        :class:`.MLE` class for maximum likelihood estimation, thus inputs to :class:`.MLE` can also be provided to
        :class:`InformationModelSelection`, as lists of length equal to the number of models.

        :param mle_estimators: A list containing a maximum-likelihood estimator (:class:`.MLE`) for each one of the
         models to be compared.
        :param criterion: Criterion to be used (:class:`.AIC`, :class:`.BIC`, :class:`.AICc)`. Default is :class:`.AIC`
        :param data: Available data. If this parameter is provided at :class:`.InformationModelSelection` object
         initialization, the model selection algorithm will be automatically performed. Alternatively, the user must
         execute the :meth:`.run` method.
        """
        self.candidate_models = [mle.inference_model for mle in mle_estimators]
        self.models_number = len(mle_estimators)
        self.criterion: InformationCriterion = criterion
        self.logger = logging.getLogger(__name__)
        self.data = data

        self.ml_estimators: list = mle_estimators
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
        if self.data is not None:
            self.run(data=self.data)

    def run(self, data: Union[list, np.ndarray]):
        """
        Run the model selection procedure, i.e. compute criterion value for all models.

        This function calls the :meth:`run` method of the :class:`.MLE` object for each model to compute the maximum
        log-likelihood, then computes the criterion value and probability for each model. If `data` are given when
        creating the :class:`.MLE` object, this method is called automatically when the object is created.

        :param data: Available data.
        """
        self.data = data

        # Loop over all the models
        for i, ml_estimator in enumerate(self.ml_estimators):
            # First evaluate ML estimate for all models, do several iterations if demanded
            ml_estimator.run(data=self.data)

            # Then minimize the criterion
            self.criterion_values[i], self.penalty_terms[i] = \
                self._minimize_info_criterion(criterion=self.criterion,
                                              data=self.data,
                                              inference_model=ml_estimator.inference_model,
                                              max_log_like=ml_estimator.max_log_like,
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
        self.ml_estimators = [self.ml_estimators[i] for i in sort_idx]
        self.criterion_values = [self.criterion_values[i] for i in sort_idx]
        self.penalty_terms = [self.penalty_terms[i] for i in sort_idx]
        self.probabilities = [self.probabilities[i] for i in sort_idx]

    @staticmethod
    def _minimize_info_criterion(
            criterion: InformationCriterion,
            data,
            inference_model,
            max_log_like,
            return_penalty=False,
    ):

        n_parameters = inference_model.n_parameters
        n_data = len(data)
        penalty_term = criterion.evaluate_criterion(n_data, n_parameters)
        if return_penalty:
            return -2 * max_log_like + penalty_term, penalty_term
        return -2 * max_log_like + penalty_term

    @staticmethod
    def _compute_probabilities(criterion_values):
        delta = np.array(criterion_values) - min(criterion_values)
        prob = np.exp(-delta / 2)
        return prob / np.sum(prob)

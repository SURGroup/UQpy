import logging
from typing import Union

from UQpy.optimization.MinimizeOptimizer import MinimizeOptimizer
from UQpy.optimization.baseclass.Optimizer import Optimizer
from beartype import beartype
from UQpy.inference.inference_models.baseclass.InferenceModel import InferenceModel
from UQpy.inference.MLE import MLE
from UQpy.utilities.ValidationTypes import RandomStateType, PositiveInteger
from UQpy.utilities.Utilities import process_random_state
from UQpy.inference.InformationTheoreticCriterion import *


class InformationModelSelection:

    # Authors: Audrey Olivier, Dimitris Giovanis
    # Last Modified: 12/19 by Audrey Olivier
    @beartype
    def __init__(
        self,
        candidate_models: list[InferenceModel],
        data: Union[list, np.ndarray],
        optimizer: Optimizer = MinimizeOptimizer(),
        criterion: InformationTheoreticCriterion = InformationTheoreticCriterion.AIC,
        random_state: RandomStateType = None,
        optimizations_number: Union[PositiveInteger, None] = None,
        initial_guess: list[np.ndarray] = None,
    ):
        """
        Perform model selection using information theoretic criteria.

        Supported criteria are :math:`BIC, AIC` (default), :math:`AICc`. This class leverages the :class:`.MLE` class
        for maximum likelihood estimation, thus inputs to :class:`.MLE` can also be provided to
        :class:`InformationModelSelection`, as lists of length equal to the number of models.

        :param candidate_models: Candidate models
        :param data: Available data
        :param optimizer: This parameter takes as input an object that implements the :class:`Optimizer` class.
         Default is the :class:`.Minimize` which utilizes the :class:`scipy.optimize.minimize` method.
        :param criterion: Criterion to be used :math:`(AIC, BIC, AICc)`. Default is :math:`AIC`
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is :any:`None`.
        :param optimizations_number: Number of iterations for the maximization procedure - see :class:`.MLE`
        :param initial_guess: Starting points for optimization - see :class:`.MLE`
        """
        if not isinstance(candidate_models, (list, tuple)) or not all(
            isinstance(model, InferenceModel) for model in candidate_models
        ):
            raise TypeError("UQpy: Input candidate_models must be a list of InferenceModel objects.")
        self.models_number = len(candidate_models)
        self.candidate_models = candidate_models
        self.data = data
        self.criterion: InformationTheoreticCriterion = criterion
        self.random_state = process_random_state(random_state)
        self.logger = logging.getLogger(__name__)

        self.optimizer = optimizer
        self.ml_estimators: list = []
        """:class:`.MLE` results for each model (contains e.g. fitted parameters)"""
        self._initialize_ml_estimators()

        # Initialize the outputs
        self.criterion_values: list = [None,] * self.models_number
        """Value of the criterion for all models."""
        self.penalty_terms: list = [None,] * self.models_number
        """Value of the penalty term for all models. Data fit term is then criterion_value - penalty_term."""
        self.probabilities: list = [None,] * self.models_number
        """Value of the model probabilities, computed as
        
        .. math:: P(M_i|d) = \dfrac{\exp(-\Delta_i/2)}{\sum_i \exp(-\Delta_i/2)}
        
        where :math:`\Delta_i = criterion_i - min_i(criterion)`"""

        # Run the model selection procedure
        if (optimizations_number is not None) or (initial_guess is not None):
            self.run(optimizations_number=optimizations_number, initial_guess=initial_guess)

    def _initialize_ml_estimators(self):
        for i, inference_model in enumerate(self.candidate_models):
            ml_estimator = MLE(
                inference_model=inference_model,
                data=self.data,
                random_state=self.random_state,
                initial_guess=None,
                optimizations_number=None,
                optimizer=self.optimizer,
            )
            self.ml_estimators.append(ml_estimator)

    def run(self, optimizations_number: PositiveInteger = 1, initial_guess=None):
        """
        Run the model selection procedure, i.e. compute criterion value for all models.

        This function calls the :meth:`run` method of the :class:`.MLE` object for each model to compute the maximum
        log-likelihood, then computes the criterion value and probability for each model.

        :param optimizations_number: Number of iterations that the optimization is run, starting at random initial
         guesses. It is only used if `initial_guess` is not provided. Default is 1. See :class:`.MLEstimation` class.
        :param initial_guess: Starting point(s) for optimization for all models. Default is :any:`None`. If not
         provided, see `optimizations_number`. See :class:`.MLE` class.
        """
        initial_guess, optimizations_number = self._check_input_data(
            initial_guess, optimizations_number
        )

        # Loop over all the models
        for i, (inference_model, ml_estimator) in enumerate(
            zip(self.candidate_models, self.ml_estimators)
        ):
            # First evaluate ML estimate for all models, do several iterations if demanded
            ml_estimator.run(
                optimizations_number=optimizations_number[i],
                initial_guess=initial_guess[i],
            )

            # Then minimize the criterion
            (
                self.criterion_values[i],
                self.penalty_terms[i],
            ) = self._minimize_info_criterion(
                criterion=self.criterion,
                data=self.data,
                inference_model=inference_model,
                max_log_like=ml_estimator.max_log_like,
                return_penalty=True,
            )

        # Compute probabilities from criterion values
        self.probabilities = self._compute_probabilities(self.criterion_values)

    def _check_input_data(self, initial_guess, optimizations_number):
        if isinstance(optimizations_number, int) or optimizations_number is None:
            optimizations_number = [optimizations_number] * self.models_number
        if not (
            isinstance(optimizations_number, list)
            and len(optimizations_number) == self.models_number
        ):
            raise ValueError(
                "UQpy: nopt should be an int or list of length models_number"
            )
        if initial_guess is None:
            initial_guess = [None] * self.models_number
        if not (
            isinstance(initial_guess, list) and len(initial_guess) == self.models_number
        ):
            raise ValueError(
                "UQpy: x0 should be a list of length models_number (or None)."
            )
        return initial_guess, optimizations_number

    def sort_models(self):
        """
        Sort models in descending order of model probability (increasing order of `criterion` value).

        This function sorts - in place - the attribute lists `candidate_models, ml_estimators, criterion_values,
        penalty_terms` and `probabilities` so that they are sorted from most probable to least probable model. It is a
        stand-alone function that is provided to help the user to easily visualize which model is the best.

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
        criterion: InformationTheoreticCriterion,
        data,
        inference_model,
        max_log_like,
        return_penalty=False,
    ):

        n_params = inference_model.parameters_number
        number_of_data = len(data)
        penalty_term = penalty_terms[criterion.name](number_of_data, n_params)
        if return_penalty:
            return -2 * max_log_like + penalty_term, penalty_term
        return -2 * max_log_like + penalty_term

    @staticmethod
    def _compute_probabilities(criterion_values):
        delta = np.array(criterion_values) - min(criterion_values)
        prob = np.exp(-delta / 2)
        return prob / np.sum(prob)

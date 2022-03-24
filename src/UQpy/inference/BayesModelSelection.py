import logging
from typing import Union

import numpy as np
from beartype import beartype

from UQpy.inference.BayesParameterEstimation import BayesParameterEstimation
from UQpy.inference.evidence_methods.HarmonicMean import HarmonicMean
from UQpy.inference.evidence_methods.baseclass import EvidenceMethod
from UQpy.inference.inference_models.baseclass.InferenceModel import InferenceModel
from UQpy.sampling import ImportanceSampling
from UQpy.utilities.ValidationTypes import PositiveInteger


class BayesModelSelection:

    # Authors: Audrey Olivier, Yuchen Zhou
    # Last modified: 01/24/2020 by Audrey Olivier
    @beartype
    def __init__(
            self,
            parameter_estimators: list[BayesParameterEstimation],
            prior_probabilities=None,
            evidence_method: EvidenceMethod = HarmonicMean(),
            nsamples: list[PositiveInteger] = None,
    ):
        """
        Perform model selection via Bayesian inference, i.e., compute model posterior probabilities given data.

        This class leverages the :class:`.BayesParameterEstimation` class to get samples from the parameter posterior
        densities. These samples are then used to compute the model evidence :code:`p(data|model)` for all models and
        the model posterior probabilities.

        :param data: Available data
        :param parameter_estimators: Parameter estimators used during the model selection algorithm.
        :param prior_probabilities: Prior probabilities of each model, default is :code:`[1/nmodels, ] * nmodels`
        :param evidence_method: as of v3, only the harmonic mean method is supported
        :param nsamples: Number of samples used in :class:`.MCMC`/:class:`.ImportanceSampling`, for each model
        """
        self.bayes_estimators: list[BayesParameterEstimation] = parameter_estimators
        """Results of the Bayesian parameter estimation."""
        self.candidate_models: list[InferenceModel] = [x.inference_model for x in self.bayes_estimators]
        """Probabilistic models used during the model selection process."""
        self.models_number = len(self.candidate_models)
        self.evidence_method = evidence_method
        self.logger = logging.getLogger(__name__)

        if prior_probabilities is None:
            self.prior_probabilities = [1.0 / len(self.candidate_models) for _ in self.candidate_models]
        else:
            self.prior_probabilities = prior_probabilities

        # Instantiate the Bayesian parameter estimators (without running them)

        self._update_bayes_estimators()

        # Initialize the outputs
        self.evidences: list = [0.0] * self.models_number
        """Value of the evidence for all models."""
        self.probabilities: list = [0.0] * self.models_number
        """Posterior probability for all models"""

        # Run the model selection procedure
        if nsamples is not None:
            self.run(nsamples=nsamples)

    def _update_bayes_estimators(self):
        for i, estimator in enumerate(self.bayes_estimators):
            if not isinstance(estimator.sampler, ImportanceSampling):
                estimator.sampler.save_log_pdf = True
                estimator.sampler.concatenate_chains = True
                estimator.sampler.dimension = estimator.inference_model.n_parameters

    @beartype
    def run(self, nsamples: Union[None, list[int]]):
        """
        Run the Bayesian model selection procedure, i.e., compute model posterior probabilities.

        This function calls the :py:meth:`run_estimation` method of the :class:`.BayesParameterEstimation` object for
        each model to sample from the parameter posterior probability, then computes the model evidence and model
        posterior probability. This function updates attributes :py:attr:`bayes_estimators`, :py:attr:`evidences` and
        :py:attr:`probabilities`. If `nsamples` are given when creating the object, this method is called
        directly when the object is created. It can also be called separately.

        :param nsamples: Number of samples used in :class:`.MCMC`/:class:`.ImportanceSampling``, for each model
        """
        self.logger.info("UQpy: Running Bayesian Model Selection.")
        # Perform mcmc for all candidate models
        for i, (inference_model, bayes_estimator) in enumerate(zip(self.candidate_models, self.bayes_estimators)):
            self.logger.info("UQpy: Running mcmc for model " + inference_model.name)
            if nsamples[i] == 0:
                continue
            bayes_estimator.run(nsamples=nsamples[i])
            self.evidences[i] = \
                self.evidence_method.estimate_evidence(inference_model=inference_model,
                                                       posterior_samples=bayes_estimator.sampler.samples,
                                                       log_posterior_values=bayes_estimator.sampler.log_pdf_values, )

        # Compute posterior probabilities
        self.probabilities = self._compute_posterior_probabilities(
            prior_probabilities=self.prior_probabilities,
            evidence_values=self.evidences)

        self.logger.info("UQpy: Bayesian Model Selection analysis completed!")

    def sort_models(self):
        """
        Sort models in descending order of model probability (increasing order of criterion value).

        This function sorts - in place - the attribute lists :py:attr:`candidate_models`, :py:attr:`probabilities`
        and :py:attr:`evidences` so that they are sorted from most probable to the least probable model. It is a
        stand-alone function that is provided to help the user to easily visualize which model is the best.

        No inputs/outputs.

        """
        sort_idx = list(np.argsort(np.array(self.probabilities)))[::-1]

        self.candidate_models = [self.candidate_models[i] for i in sort_idx]
        self.prior_probabilities = [self.prior_probabilities[i] for i in sort_idx]
        self.probabilities = [self.probabilities[i] for i in sort_idx]
        self.evidences = [self.evidences[i] for i in sort_idx]

    @staticmethod
    def _compute_posterior_probabilities(prior_probabilities, evidence_values):
        """
        Compute the model probability given prior probabilities P(M) and evidence values p(data|M).

        Model posterior probability P(M|data) is proportional to p(data|M)P(M). Posterior probabilities sum up to 1 over
        all models. This function is a utility function (static method), called within the run_estimation method.

        **Inputs:**

        :param prior_probabilities: Values of prior probabilities for all models.
        :type prior_probabilities: list (length nmodels) of floats

        :param prior_probabilities: Values of evidence for all models.
        :type prior_probabilities: list (length nmodels) of floats

        **Output/Returns:**

        :return probabilities: Values of model posterior probabilities
        :rtype probabilities: list (length nmodels) of floats

        """
        scaled_evidences = [evidence * prior_probability for (evidence, prior_probability)
                            in zip(evidence_values, prior_probabilities)]
        return scaled_evidences / np.sum(scaled_evidences)

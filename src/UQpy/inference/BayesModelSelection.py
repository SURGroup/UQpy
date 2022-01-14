import logging
from typing import Union, List

import numpy as np
from beartype import beartype

from UQpy.inference.BayesParameterEstimation import BayesParameterEstimation
from UQpy.inference.evidence_methods.HarmonicMean import HarmonicMean
from UQpy.inference.evidence_methods.baseclass import EvidenceMethod
from UQpy.inference.inference_models.baseclass.InferenceModel import InferenceModel
from UQpy.sampling import ImportanceSampling
from UQpy.sampling.mcmc.baseclass import MCMC
from UQpy.utilities.ValidationTypes import PositiveInteger


class BayesModelSelection:

    # Authors: Audrey Olivier, Yuchen Zhou
    # Last modified: 01/24/2020 by Audrey Olivier
    @beartype
    def __init__(
            self,
            candidate_models: list[InferenceModel],
            data,
            sampling_class: list[Union[ImportanceSampling, MCMC]],
            prior_probabilities=None,
            evidence_method: EvidenceMethod = HarmonicMean(),
            nsamples: list[PositiveInteger] = None,
            nsamples_per_chain: list[PositiveInteger] = None,
    ):
        """
        Perform model selection via Bayesian inference, i.e., compute model posterior probabilities given data.

        This class leverages the :class:`.BayesParameterEstimation` class to get samples from the parameter posterior
        densities. These samples are then used to compute the model evidence `p(data|model)` for all models and the
        model posterior probabilities.

        :param candidate_models: Candidate models
        :param data: Available data
        :param sampling_class: List of class instances, that implement the of :class:`.SamplingInput` abstract
         class
        :param prior_probabilities: Prior probabilities of each model, default is :code:`[1/nmodels, ] * nmodels`
        :param evidence_method: as of v3, only the harmonic mean method is supported
        :param nsamples: Number of samples used in :class:`.MCMC`/:class:`.ImportanceSampling`, for each model
        :param nsamples_per_chain: Number of samples per chain used in :class:`.MCMC`, for each model
        """
        self.candidate_models = candidate_models
        self.models_number = len(candidate_models)
        self.data = data
        self.evidence_method = evidence_method
        self.sampling_classes = sampling_class
        self.logger = logging.getLogger(__name__)

        if prior_probabilities is None:
            self.prior_probabilities = [1.0 / len(candidate_models) for _ in candidate_models]
        else:
            self.prior_probabilities = prior_probabilities

        # Instantiate the Bayesian parameter estimators (without running them)
        self.bayes_estimators: list = []
        """Results of the Bayesian parameter estimation."""
        self._create_bayes_estimators(candidate_models, sampling_class)

        # Initialize the outputs
        self.evidences: list = [0.0] * self.models_number
        """Value of the evidence for all models."""
        self.probabilities: list = [0.0] * self.models_number
        """Posterior probability for all models"""

        # Run the model selection procedure
        if nsamples is not None or nsamples_per_chain is not None:
            self.run(nsamples=nsamples,
                     nsamples_per_chain=nsamples_per_chain, )

    def _create_bayes_estimators(self, candidate_models, sampling_classes):
        if len(candidate_models) != len(sampling_classes):
            raise TypeError(
                "UQpy: The number of sampling_classes provided must be equal to the "
                "number of candidate_models")
        for i, inference_model in enumerate(self.candidate_models):
            sampling = sampling_classes[i]
            # sampling_input.random_state = self.random_state
            if not isinstance(sampling_classes, ImportanceSampling):
                sampling.save_log_pdf = True
                sampling.concatenate_chains = True
                sampling.dimension = inference_model.n_parameters

            bayes_estimator = BayesParameterEstimation(sampling_class=sampling, inference_model=inference_model,
                                                       data=self.data)
            self.bayes_estimators.append(bayes_estimator)

    @beartype
    def run(self,
            nsamples: Union[None, list[PositiveInteger]] = None,
            nsamples_per_chain: Union[None, list[PositiveInteger]] = None, ):
        """
        Run the Bayesian model selection procedure, i.e., compute model posterior probabilities.

        This function calls the :py:meth:`run_estimation` method of the :class:`.BayesParameterEstimation` object for
        each model to sample from the parameter posterior probability, then computes the model evidence and model
        posterior probability. This function updates attributes `bayes_estimators`, `evidences` and `probabilities`. If
        `nsamples` or `nsamples_per_chain` are given when creating the object, this method is called
        directly when the object is created. It can also be called separately.

        :param nsamples: Number of samples used in :class:`.MCMC`/:class:`.ImportanceSampling``, for each model
        :param nsamples_per_chain: Number of samples per chain used in :class:`.MCMC`, for each model
        """
        self.logger.info("UQpy: Running Bayesian Model Selection.")
        # Perform mcmc for all candidate models
        for i, (inference_model, bayes_estimator) in enumerate(
                zip(self.candidate_models, self.bayes_estimators)):
            self.logger.info("UQpy: Running mcmc for model " + inference_model.name)
            if nsamples is not None:
                bayes_estimator.run(nsamples=nsamples[i])
            elif nsamples_per_chain is not None:
                bayes_estimator.run(nsamples_per_chain=nsamples_per_chain[i])
            else:
                raise ValueError(
                    "UQpy: either nsamples or nsamples_per_chain should be non None")
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

        This function sorts - in place - the attribute lists `candidate_models`, `prior_probabilities`, `probabilities`
        and `evidences` so that they are sorted from most probable to least probable model. It is a stand-alone function
        that is provided to help the user to easily visualize which model is the best.

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

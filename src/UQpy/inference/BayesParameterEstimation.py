import logging
from typing import Union, Annotated
from beartype import beartype
from beartype.vale import Is

from UQpy.sampling.input_data import SamplingInput
from UQpy.sampling.input_data.ISInput import ISInput
from UQpy.utilities.NoPublicConstructor import NoPublicConstructor
from UQpy.utilities.ValidationTypes import PositiveInteger
from UQpy.inference.inference_models.baseclass.InferenceModel import InferenceModel
from UQpy.sampling import MCMC, ImportanceSampling

McmcInput = Annotated[SamplingInput, Is[lambda input: not isinstance(input, ISInput)]]


class BayesParameterEstimation(metaclass=NoPublicConstructor):
    # Authors: Audrey Olivier, Dimitris Giovanis
    # Last Modified: 12/19 by Audrey Olivier
    @beartype
    def __init__(
        self,
        inference_model: InferenceModel,
        data,
        sampling_class: Union[MCMC, ImportanceSampling] = None,
        samples_number: Union[None, int] = None,
        samples_number_per_chain: Union[None, int] = None,
    ):
        """
        Estimate the parameter posterior density given some data.

        This class generates samples from the parameter posterior distribution using Markov Chain Monte Carlo or
        Importance Sampling. It leverages the :class:`.MCMC` and :class:`.ImportanceSampling` classes from the
        :py:mod:`.sampling` module.

        :param inference_model: The inference model that defines the likelihood function.
        :param data: Available data, `ndarray` of shape consistent with log-likelihood function in
         :class:`.InferenceModel`
        :param sampling_class: Class instance, must be a subclass of :class:`.MCMC` or :class:`.ImportanceSampling`.
        :param samples_number: Number of samples used in MCMC/IS, see :meth:`run` method.
        :param samples_number_per_chain: Number of samples per chain used in mcmc, see `run` method.
        """
        self.inference_model = inference_model
        self.data = data
        self.logger = logging.getLogger(__name__)
        self.sampler = sampling_class
        """Sampling method object, contains e.g. the posterior samples.
        This object is created along with the :class:`.BayesParameterEstimation` object, and its `run` method is called
        whenever the `run` method of the :class:`.BayesParameterEstimation` is called."""
        if (samples_number is not None) or (samples_number_per_chain is not None):
            self.run(
                samples_number=samples_number,
                samples_number_per_chain=samples_number_per_chain,
            )

    @classmethod
    @beartype
    def create_with_mcmc_sampling(
        cls,
        mcmc_input: McmcInput,
        inference_model: InferenceModel,
        data,
        samples_number: int = None,
        samples_number_per_chain: Union[None, int] = None,
    ):
        """
        One of the two possible ways to create a :class:`BayesParameterEstimation` object when the user wants to use
        MCMC sampling.

        :param mcmc_input: Class instance, must be a class of :class:`.SamplingInput` used by the MCMC algorithms
        :param inference_model: The inference model that defines the likelihood function.
        :param data: Available data, `ndarray` of shape consistent with log-likelihood function in
         :class:`.InferenceModel`
        :param samples_number: Number of samples used in MCMC, see :meth:`run` method.
        :param samples_number_per_chain: Number of samples per chain used in mcmc, see `run` method.
        """
        class_type = type(mcmc_input)
        sampling_class = SamplingInput.input_to_class[class_type]
        if mcmc_input.seed is None:
            if inference_model.prior is None or not hasattr(
                inference_model.prior, "rvs"
            ):
                raise ValueError(
                    "UQpy: A prior with a rvs method must be provided for the InferenceModel"
                    " or a seed must be provided for MCMC."
                )
            else:
                mcmc_input.seed = inference_model.prior.rvs(
                    nsamples=mcmc_input.chains_number,
                    random_state=mcmc_input.random_state,
                )
        mcmc_input.log_pdf_target = inference_model.evaluate_log_posterior
        mcmc_input.args_target = (data,)
        sampler = sampling_class(mcmc_input)
        return cls._create(
            inference_model, data, sampler, samples_number, samples_number_per_chain
        )

    @classmethod
    @beartype
    def create_with_importance_sampling(
        cls,
        inference_model: InferenceModel,
        data,
        is_input: ISInput,
        samples_number: int = None,
    ):
        """
        The second alternative to create a :class:`BayesParameterEstimation` object when the user wants to use
        ImpostanceSampling sampling.

        :param inference_model: The inference model that defines the likelihood function.
        :param data: Available data, `ndarray` of shape consistent with log-likelihood function in
         :class:`.InferenceModel`
        :param is_input: Class instance, must be a class of :class:`.SamplingInput` used by the ImportanceSampling
         algorithm
        :param samples_number: Number of samples used in IS, see :meth:`run` method.
        """
        if is_input.proposal is None:
            if inference_model.prior is None:
                raise NotImplementedError(
                    "UQpy: A proposal density of the ImportanceSampling"
                    " or a prior to the Inference model  must be provided."
                )
            is_input.proposal = inference_model.prior
        is_input.log_pdf_target = inference_model.evaluate_log_posterior
        is_input.args_target = (data,)
        sampler = ImportanceSampling(is_input=is_input)
        return cls._create(inference_model, data, sampler, samples_number)


    sampling_actions = {
        MCMC: lambda sampler, nsamples, nsamples_per_chain: sampler.run(
            samples_number=nsamples, samples_number_per_chain=nsamples_per_chain
        ),
        ImportanceSampling: lambda sampler, nsamples, nsamples_per_chain: sampler.run(
            samples_number=nsamples
        ),
    }

    @beartype
    def run(
        self, samples_number: PositiveInteger = None, samples_number_per_chain=None
    ):
        """
        Run the Bayesian inference procedure, i.e., sample from the parameter posterior distribution.

        This function calls the :meth:`run` method of the `sampler` attribute to generate samples from the parameter
        posterior distribution.

        :param samples_number: Number of samples used in :class:`.MCMC`/:class:`.ImportanceSampling`
        :param samples_number_per_chain: Number of samples per chain used in :class:`.MCMC`
        """
        method = MCMC if isinstance(self.sampler, MCMC) else ImportanceSampling
        BayesParameterEstimation.sampling_actions[method](
            self.sampler, samples_number, samples_number_per_chain
        )

        self.logger.info(
            "UQpy: Parameter estimation with "
            + self.sampler.__class__.__name__
            + " completed successfully!"
        )

import logging
from typing import Union

from beartype import beartype
from UQpy.utilities.ValidationTypes import PositiveInteger
from UQpy.inference.inference_models.baseclass.InferenceModel import InferenceModel
from UQpy.sampling import MCMC, ImportanceSampling


class BayesParameterEstimation:
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
        self._method = MCMC if isinstance(self.sampler, MCMC) else ImportanceSampling
        if (samples_number is not None) or (samples_number_per_chain is not None):
            self.run(samples_number=samples_number, samples_number_per_chain=samples_number_per_chain,)

    sampling_actions = {
        MCMC: lambda sampler, nsamples, nsamples_per_chain: sampler.run(
            samples_number=nsamples, samples_number_per_chain=nsamples_per_chain),
        ImportanceSampling: lambda sampler, nsamples, nsamples_per_chain: sampler.run(samples_number=nsamples),
    }

    @beartype
    def run(self, samples_number: PositiveInteger = None, samples_number_per_chain=None):
        """
        Run the Bayesian inference procedure, i.e., sample from the parameter posterior distribution.

        This function calls the :meth:`run` method of the `sampler` attribute to generate samples from the parameter
        posterior distribution.

        :param samples_number: Number of samples used in :class:`.MCMC`/:class:`.ImportanceSampling`
        :param samples_number_per_chain: Number of samples per chain used in :class:`.MCMC`
        """

        BayesParameterEstimation.sampling_actions[self._method](self.sampler, samples_number, samples_number_per_chain)

        self.logger.info("UQpy: Parameter estimation with " + self.sampler.__class__.__name__
                         + " completed successfully!")

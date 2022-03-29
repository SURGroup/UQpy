import logging
from typing import Union

import numpy as np
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
        data: Union[list, np.ndarray],
        sampling_class: Union[MCMC, ImportanceSampling] = None,
        nsamples: Union[None, int] = None,
    ):
        """
        Estimate the parameter posterior density given some data.

        This class generates samples from the parameter posterior distribution using Markov Chain Monte Carlo or
        Importance Sampling. It leverages the :class:`.MCMC` and :class:`.ImportanceSampling` classes from the
        :py:mod:`.sampling` module.

        :param inference_model: The inference model that defines the likelihood function.
        :param data: Available data, :class:`numpy.ndarray` of shape consistent with log-likelihood function in
         :class:`.InferenceModel`
        :param sampling_class: Class instance, must be a subclass of :class:`.MCMC` or :class:`.ImportanceSampling`.
        :param nsamples: Number of samples used in :class:`.MCMC`/:class:`ImportanceSampling`, see
         :meth:`run` method. If the `nsamples` parameter is provided then :class:`.BayesParameterEstimation` is
         automatically performed. In case an :class:`.ImportanceSampling` method is used to perform the parameter
         estimation, then `nsamples` equal to the total number of samples. In case an :class:`.MCMC` sampler is used,
         and the given `nsamples` is not a multiple of `n_chains`, then `nsamples` is set to the next largest integer
         that is a multiple of  `nchains`.
        """
        self.inference_model = inference_model
        self.data = data
        self.logger = logging.getLogger(__name__)
        self.sampler: Union[MCMC, ImportanceSampling] = sampling_class
        """Sampling method object, contains e.g. the posterior samples.

        This must be created along side the :class:`.BayesParameterEstimation` object, and its run method is called 
        whenever the :py:meth:`run` method of the :class:`.BayesParameterEstimation` is called.
        """
        if isinstance(self.sampler, ImportanceSampling):
            if self.sampler.proposal is None:
                if inference_model.prior is None:
                    raise NotImplementedError(
                        "UQpy: A proposal density of the ImportanceSampling"
                        " or a prior to the Inference model  must be provided.")
                self.sampler.proposal = inference_model.prior
            self.sampler._args_target = (data,)
            self.sampler.log_pdf_target = inference_model.evaluate_log_posterior
        elif isinstance(self.sampler, MCMC):
            if self.sampler._initialization_seed is None:
                if inference_model.prior is None or not hasattr(inference_model.prior, "rvs"):
                    raise ValueError(
                        "UQpy: A prior with a rvs method must be provided for the InferenceModel"
                        " or a seed must be provided for MCMC.")
                else:
                    self.sampler.seed = inference_model.prior.rvs(nsamples=self.sampler.n_chains,
                                                                  random_state=self.sampler.random_state, ).tolist()
            self.sampler.log_pdf_target = inference_model.evaluate_log_posterior
            self.sampler.pdf_target = None
            self.sampler.args_target = (data,)
            (self.sampler.evaluate_log_target,
             self.sampler.evaluate_log_target_marginals,) = \
                self.sampler._preprocess_target(pdf_=None,
                                                log_pdf_=self.sampler.log_pdf_target,
                                                args=self.sampler.args_target)

        if nsamples is not None:
            self.run(nsamples=nsamples)

    @beartype
    def run(self, nsamples: PositiveInteger = None):
        """
        Run the Bayesian inference procedure, i.e., sample from the parameter posterior distribution.

        This function calls the :meth:`run` method of the :py:attr:`.sampler` attribute to generate samples from the
        parameter posterior distribution.

        :param nsamples: Number of samples used in :class:`.MCMC`/:class:`.ImportanceSampling`.  In case an
         :class:`.ImportanceSampling` method is used to perform the parameter estimation, then `nsamples` equal to the
         total number of samples. In case an :class:`.MCMC` sampler is used, and the given `nsamples` is not a multiple
         of `n_chains`, then `nsamples` is set to the next largest integer that is a multiple of  `nchains`.
        """
        self.sampler.run(nsamples=nsamples)

        self.logger.info("UQpy: Parameter estimation with " + self.sampler.__class__.__name__
                         + " completed successfully!")

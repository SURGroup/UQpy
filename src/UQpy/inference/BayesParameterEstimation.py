import logging
from typing import Union
from beartype import beartype
from UQpy.sampling.input_data.ISInput import ISInput
from UQpy.sampling.mcmc import *
from UQpy.sampling.input_data.DramInput import DramInput
from UQpy.sampling.input_data.DreamInput import DreamInput
from UQpy.sampling.input_data.MhInput import MhInput
from UQpy.sampling.input_data.MmhInput import MmhInput
from UQpy.sampling.input_data.StretchInput import StretchInput
from UQpy.utilities.NoPublicConstructor import NoPublicConstructor
from UQpy.utilities.ValidationTypes import PositiveInteger
from UQpy.inference.inference_models.baseclass.InferenceModel import InferenceModel
from UQpy.sampling import MCMC, ImportanceSampling, MetropolisHastings


class BayesParameterEstimation(metaclass=NoPublicConstructor):
    """
    Estimate the parameter posterior density given some data.

    This class generates samples from the parameter posterior distribution using Markov Chain Monte Carlo or Importance
    Sampling. It leverages the ``mcmc`` and ``IS`` classes from the ``sampling`` module.


    **Inputs:**

    * **inference_model** (object of class ``InferenceModel``):
        The inference model that defines the likelihood function.

    * **data** (`ndarray`):
        Available data, `ndarray` of shape consistent with log-likelihood function in ``InferenceModel``

    * **sampling_class** (class instance):
        Class instance, must be a subclass of ``mcmc`` or ``IS``.

    * **kwargs_sampler**:
        Keyword arguments of the sampling class, see ``sampling.mcmc`` or ``sampling.IS``.

        Note on the seed for ``mcmc``: if input `seed` is not provided, a seed (`ndarray` of shape
        `(nchains, dimension)`) is sampled from the prior pdf, which must have an `rvs` method.

        Note on the proposal for ``IS``: if no input `proposal` is provided, the prior is used as proposal.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **nsamples** (`int`):
        Number of samples used in mcmc/IS, see `run` method.

    * **samples_per_chain** (`int`):
        Number of samples per chain used in mcmc, see `run` method.

    If both `nsamples` and `nsamples_per_chain` are `None`, the object is created but the sampling procedure is not run,
    one must call the ``run`` method.

    **Attributes:**

    * **sampler** (object of ``sampling`` class specified by `sampling_class`):
        Sampling method object, contains e.g. the posterior samples.

        This object is created along with the ``BayesParameterEstimation`` object, and its `run` method is called
        whenever the `run` method of the ``BayesParameterEstimation`` is called.

    **Methods:**

    """

    # Authors: Audrey Olivier, Dimitris Giovanis
    # Last Modified: 12/19 by Audrey Olivier
    @beartype
    def __init__(self,
                 inference_model: InferenceModel, data,
                 sampling_class: Union[MCMC, ImportanceSampling] = None,
                 samples_number: Union[None, int] = None,
                 samples_number_per_chain: Union[None, int] = None):

        self.inference_model = inference_model
        self.data = data
        self.logger = logging.getLogger(__name__)
        self.sampler = sampling_class
        if (samples_number is not None) or (samples_number_per_chain is not None):
            self.run(samples_number=samples_number, samples_number_per_chain=samples_number_per_chain)

    @classmethod
    @beartype
    def create_with_mcmc_sampling(cls,
                                  mcmc_input: Union[DramInput, DreamInput, MhInput, MmhInput, StretchInput],
                                  inference_model: InferenceModel,
                                  data,
                                  samples_number: int = None,
                                  samples_number_per_chain: Union[None, int] = None):
        class_type = type(mcmc_input)
        sampling_class = BayesParameterEstimation.input_to_class[class_type]
        if mcmc_input.seed is None:
            if inference_model.prior is None or not hasattr(inference_model.prior, 'rvs'):
                raise ValueError('UQpy: A prior with a rvs method must be provided for the InferenceModel'
                                 ' or a seed must be provided for MCMC.')
            else:
                mcmc_input.seed = inference_model.prior.rvs(nsamples=mcmc_input.chains_number,
                                                            random_state=mcmc_input.random_state)
        mcmc_input.log_pdf_target = inference_model.evaluate_log_posterior
        mcmc_input.args_target = (data, )
        sampler = sampling_class(mcmc_input)
        return cls._create(inference_model, data, sampler, samples_number, samples_number_per_chain)

    @classmethod
    @beartype
    def create_with_importance_sampling(cls,
                                        inference_model: InferenceModel,
                                        data,
                                        is_input: ISInput,
                                        samples_number: int = None):
        if is_input.proposal is None:
            if inference_model.prior is None:
                raise NotImplementedError('UQpy: A proposal density of the ImportanceSampling'
                                          ' or a prior to the Inference model  must be provided.')
            is_input.proposal = inference_model.prior
        is_input.log_pdf_target = inference_model.evaluate_log_posterior
        is_input.args_target = (data,)
        sampler = ImportanceSampling(is_input=is_input)
        return cls._create(inference_model, data, sampler, samples_number)

    input_to_class = {
        DramInput: DRAM,
        DreamInput: DREAM,
        MhInput: MetropolisHastings,
        MmhInput: ModifiedMetropolisHastings,
        StretchInput: Stretch
    }

    sampling_actions = {
        MCMC: lambda sampler, nsamples, nsamples_per_chain:
        sampler.run(samples_number=nsamples, samples_number_per_chain=nsamples_per_chain),
        ImportanceSampling: lambda sampler, nsamples, nsamples_per_chain:
        sampler.run(samples_number=nsamples)
    }

    @beartype
    def run(self, samples_number: PositiveInteger = None, samples_number_per_chain=None):
        """
        Run the Bayesian inference procedure, i.e., sample from the parameter posterior distribution.

        This function calls the ``run`` method of the `sampler` attribute to generate samples from the parameter
        posterior distribution.

        **Inputs:**

        * **nsamples** (`int`):
            Number of samples used in ``mcmc``/``IS``

        * **samples_per_chain** (`int`):
            Number of samples per chain used in ``mcmc``

        """
        method = MCMC if isinstance(self.sampler, MCMC) else ImportanceSampling
        BayesParameterEstimation.sampling_actions[method](self.sampler, samples_number, samples_number_per_chain)

        self.logger.info('UQpy: Parameter estimation with '
                         + self.sampler.__class__.__name__ + ' completed successfully!')

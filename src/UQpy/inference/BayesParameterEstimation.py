import numpy as np

from UQpy.inference.inference_models.baseclass.InferenceModel import InferenceModel
from UQpy.sampling import MCMC, ImportanceSampling,MetropolisHastings


class BayesParameterEstimation:
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
    def __init__(self, inference_model, data, sampling_class=None, nsamples=None, nsamples_per_chain=None,
                 random_state=None, verbose=False):

        self.inference_model = inference_model
        if not isinstance(self.inference_model, InferenceModel):
            raise TypeError('UQpy: Input inference_model should be of type InferenceModel')
        self.data = data
        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')
        self.verbose = verbose

        if not issubclass(sampling_class, MCMC) or not issubclass(sampling_class, ImportanceSampling):
            raise ValueError('UQpy: Sampling_class should be either a MCMC algorithm or IS.')
        self.sampler = sampling_class

        # Run the analysis if a certain number of samples was provided
        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

    sampling_actions = {
        MCMC: lambda sampler, nsamples, nsamples_per_chain:
            sampler.run(number_of_samples=nsamples, nsamples_per_chain=nsamples_per_chain),
        ImportanceSampling: lambda sampler, nsamples, nsamples_per_chain:
            sampler.run(nsamples=nsamples)
    }
    #
    # @classmethod
    # def create_with_mcmc_sampling(cls, inference_model, mcmc_class=MetropolisHastings):
    #     if 'seed' not in kwargs_sampler.keys() or kwargs_sampler['seed'] is None:
    #         if inference_model.prior is None or not hasattr(inference_model.prior, 'rvs'):
    #             raise NotImplementedError('UQpy: A prior with a rvs method or a seed must be provided for MCMC.')
    #         else:
    #             kwargs_sampler['seed'] = self.inference_model.prior.rvs(
    #                 nsamples=kwargs_sampler['nchains'], random_state=self.random_state)
    #     sampling_class = sampling_class(
    #         dimension=self.inference_model.nparams, verbose=self.verbose, random_state=self.random_state,
    #         log_pdf_target=self.inference_model.evaluate_log_posterior, args_target=(self.data,),
    #         nsamples=None, nsamples_per_chain=None, **kwargs_sampler)

    def run(self, nsamples=None, nsamples_per_chain=None):
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

        BayesParameterEstimation.sampling_actions[self.sampler](self.sampler, nsamples, nsamples_per_chain)

        if self.verbose:
            print('UQpy: Parameter estimation with ' + self.sampler.__class__.__name__ + ' completed successfully!')





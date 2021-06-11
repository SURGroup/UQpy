import numpy as np

from UQpy.Inference.InferenceModel import InferenceModel
from UQpy.SampleMethods import MCMC, IS


########################################################################################################################
########################################################################################################################
#                                  Bayesian Parameter estimation
########################################################################################################################

class BayesParameterEstimation:
    """
    Estimate the parameter posterior density given some data.

    This class generates samples from the parameter posterior distribution using Markov Chain Monte Carlo or Importance
    Sampling. It leverages the ``MCMC`` and ``IS`` classes from the ``SampleMethods`` module.


    **Inputs:**

    * **inference_model** (object of class ``InferenceModel``):
        The inference model that defines the likelihood function.

    * **data** (`ndarray`):
        Available data, `ndarray` of shape consistent with log-likelihood function in ``InferenceModel``

    * **sampling_class** (class instance):
        Class instance, must be a subclass of ``MCMC`` or ``IS``.

    * **kwargs_sampler**:
        Keyword arguments of the sampling class, see ``SampleMethods.MCMC`` or ``SampleMethods.IS``.

        Note on the seed for ``MCMC``: if input `seed` is not provided, a seed (`ndarray` of shape
        `(nchains, dimension)`) is sampled from the prior pdf, which must have an `rvs` method.

        Note on the proposal for ``IS``: if no input `proposal` is provided, the prior is used as proposal.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **nsamples** (`int`):
        Number of samples used in MCMC/IS, see `run` method.

    * **samples_per_chain** (`int`):
        Number of samples per chain used in MCMC, see `run` method.

    If both `nsamples` and `nsamples_per_chain` are `None`, the object is created but the sampling procedure is not run,
    one must call the ``run`` method.

    **Attributes:**

    * **sampler** (object of ``SampleMethods`` class specified by `sampling_class`):
        Sampling method object, contains e.g. the posterior samples.

        This object is created along with the ``BayesParameterEstimation`` object, and its `run` method is called
        whenever the `run` method of the ``BayesParameterEstimation`` is called.

    **Methods:**

    """
    # Authors: Audrey Olivier, Dimitris Giovanis
    # Last Modified: 12/19 by Audrey Olivier
    def __init__(self, inference_model, data, sampling_class=None, nsamples=None, nsamples_per_chain=None,
                 random_state=None, verbose=False, **kwargs_sampler):

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

        from UQpy.SampleMethods import MCMC, IS
        # MCMC algorithm
        if issubclass(sampling_class, MCMC):
            # If the seed is not provided, sample one from the prior pdf of the parameters
            if 'seed' not in kwargs_sampler.keys() or kwargs_sampler['seed'] is None:
                if self.inference_model.prior is None or not hasattr(self.inference_model.prior, 'rvs'):
                    raise NotImplementedError('UQpy: A prior with a rvs method or a seed must be provided for MCMC.')
                else:
                    kwargs_sampler['seed'] = self.inference_model.prior.rvs(
                        nsamples=kwargs_sampler['nchains'], random_state=self.random_state)
            self.sampler = sampling_class(
                dimension=self.inference_model.nparams, verbose=self.verbose, random_state=self.random_state,
                log_pdf_target=self.inference_model.evaluate_log_posterior, args_target=(self.data, ),
                nsamples=None, nsamples_per_chain=None, **kwargs_sampler)

        elif issubclass(sampling_class, IS):
            # Importance distribution is either given by the user, or it is set as the prior of the model
            if 'proposal' not in kwargs_sampler or kwargs_sampler['proposal'] is None:
                if self.inference_model.prior is None:
                    raise NotImplementedError('UQpy: A proposal density or a prior must be provided.')
                kwargs_sampler['proposal'] = self.inference_model.prior

            self.sampler = sampling_class(
                log_pdf_target=self.inference_model.evaluate_log_posterior, args_target=(self.data, ),
                random_state=self.random_state, verbose=self.verbose, nsamples=None, **kwargs_sampler)

        else:
            raise ValueError('UQpy: Sampling_class should be either a MCMC algorithm or IS.')

        # Run the analysis if a certain number of samples was provided
        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

    def run(self, nsamples=None, nsamples_per_chain=None):
        """
        Run the Bayesian inference procedure, i.e., sample from the parameter posterior distribution.

        This function calls the ``run`` method of the `sampler` attribute to generate samples from the parameter
        posterior distribution.

        **Inputs:**

        * **nsamples** (`int`):
            Number of samples used in ``MCMC``/``IS``

        * **samples_per_chain** (`int`):
            Number of samples per chain used in ``MCMC``

        """

        if isinstance(self.sampler, MCMC):
            self.sampler.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

        elif isinstance(self.sampler, IS):
            if nsamples_per_chain is not None:
                raise ValueError('UQpy: nsamples_per_chain is not an appropriate input for IS.')
            self.sampler.run(nsamples=nsamples)

        else:
            raise ValueError('UQpy: sampling class should be a subclass of MCMC or IS')

        if self.verbose:
            print('UQpy: Parameter estimation with ' + self.sampler.__class__.__name__ + ' completed successfully!')

import numpy as np

from UQpy.Inference.BayesParameterEstimation import BayesParameterEstimation
from UQpy.Inference.InferenceModel import InferenceModel


########################################################################################################################
########################################################################################################################
#                                  Bayesian Model Selection
########################################################################################################################


class BayesModelSelection:

    """
    Perform model selection via Bayesian inference, i.e., compute model posterior probabilities given data.

    This class leverages the ``BayesParameterEstimation`` class to get samples from the parameter posterior densities.
    These samples are then used to compute the model evidence `p(data|model)` for all models and the model posterior
    probabilities.

    **References:**

    1. A.E. Raftery, M.A. Newton, J.M. Satagopan, and P.N. Krivitsky. "Estimating the integrated likelihood via
       posterior simulation using the harmonic mean identity". In Bayesian Statistics 8, pages 1–45, 2007.

    **Inputs:**

    * **candidate_models** (`list` of ``InferenceModel`` objects):
        Candidate models

    * **data** (`ndarray`):
        Available data

    * **prior_probabilities** (`list` of `floats`):
        Prior probabilities of each model, default is [1/nmodels, ] * nmodels

    * **method_evidence_computation** (`str`):
        as of v3, only the harmonic mean method is supported

    * **kwargs**:
        Keyword arguments to the ``BayesParameterEstimation`` class, for each model.

        Keys must refer to names of inputs to the ``MLEstimation`` class, and values should be lists of length
        `nmodels`, ordered in the same way as input candidate_models. For example, setting
        `kwargs={`sampling_class': [MH, Stretch]}` means that the MH algorithm will be used for sampling from the
        parameter posterior pdf of the 1st candidate model, while the Stretch algorithm will be used for the 2nd model.

    * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
        Random seed used to initialize the pseudo-random number generator. Default is None.

        If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
        object itself can be passed directly.

    * **nsamples** (`list` of `int`):
        Number of samples used in ``MCMC``/``IS``, for each model

    * **samples_per_chain** (`list` of `int`):
        Number of samples per chain used in ``MCMC``, for each model

    If `nsamples` and `nsamples_per_chain` are both `None`, the object is created but the model selection procedure is
    not run, one must then call the ``run`` method.

    **Attributes:**

    * **bayes_estimators** (`list` of ``BayesParameterEstimation`` objects):
        Results of the Bayesian parameter estimation

    * **self.evidences** (`list` of `floats`):
        Value of the evidence for all models

    * **probabilities** (`list` of `floats`):
        Posterior probability for all models

    **Methods:**

    """
    # Authors: Audrey Olivier, Yuchen Zhou
    # Last modified: 01/24/2020 by Audrey Olivier
    def __init__(self, candidate_models, data, prior_probabilities=None, method_evidence_computation='harmonic_mean',
                 random_state=None, verbose=False, nsamples=None, nsamples_per_chain=None, **kwargs):

        # Check inputs: candidate_models is a list of instances of Model, data must be provided, and input arguments
        # for MCMC must be provided as a list of length len(candidate_models)
        if (not isinstance(candidate_models, list)) or (not all(isinstance(model, InferenceModel)
                                                                for model in candidate_models)):
            raise TypeError('UQpy: A list InferenceModel objects must be provided.')
        self.candidate_models = candidate_models
        self.nmodels = len(candidate_models)
        self.data = data
        self.method_evidence_computation = method_evidence_computation
        self.random_state = random_state
        if isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)
        elif not isinstance(self.random_state, (type(None), np.random.RandomState)):
            raise TypeError('UQpy: random_state must be None, an int or an np.random.RandomState object.')
        self.verbose = verbose

        if prior_probabilities is None:
            self.prior_probabilities = [1. / len(candidate_models) for _ in candidate_models]
        else:
            self.prior_probabilities = prior_probabilities

        # Instantiate the Bayesian parameter estimators (without running them)
        self.bayes_estimators = []
        if not all(isinstance(value, (list, tuple)) for (key, value) in kwargs.items()) or not all(
                len(value) == len(candidate_models) for (key, value) in kwargs.items()):
            raise TypeError('UQpy: Extra inputs to model selection must be lists of length len(candidate_models)')
        for i, inference_model in enumerate(self.candidate_models):
            kwargs_i = dict([(key, value[i]) for (key, value) in kwargs.items()])
            kwargs_i.update({'concat_chains': True, 'save_log_pdf': True})
            bayes_estimator = BayesParameterEstimation(
                inference_model=inference_model, data=self.data, verbose=self.verbose,
                random_state=self.random_state, nsamples=None, nsamples_per_chain=None, **kwargs_i)
            self.bayes_estimators.append(bayes_estimator)

        # Initialize the outputs
        self.evidences = [0.] * self.nmodels
        self.probabilities = [0.] * self.nmodels

        # Run the model selection procedure
        if nsamples is not None or nsamples_per_chain is not None:
            self.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

    def run(self, nsamples=None, nsamples_per_chain=None):
        """
        Run the Bayesian model selection procedure, i.e., compute model posterior probabilities.

        This function calls the ``run_estimation`` method of the ``BayesParameterEstimation`` object for each model to
        sample from the parameter posterior probability, then computes the model evidence and model posterior
        probability. This function updates attributes `bayes_estimators`, `evidences` and `probabilities`. If `nsamples`
        or `nsamples_per_chain` are given when creating the object, this method is called directly when the object is
        created. It can also be called separately.

        **Inputs:**

        * **nsamples** (`list` of `int`):
            Number of samples used in ``MCMC``/``IS``, for each model

        * **samples_per_chain** (`list` of `int`):
            Number of samples per chain used in ``MCMC``, for each model

        """

        if nsamples is not None and not (isinstance(nsamples, list) and len(nsamples) == self.nmodels
                                         and all(isinstance(n, int) for n in nsamples)):
            raise ValueError('UQpy: nsamples should be a list of integers')
        if nsamples_per_chain is not None and not (isinstance(nsamples_per_chain, list)
                                                   and len(nsamples_per_chain) == self.nmodels
                                                   and all(isinstance(n, int) for n in nsamples_per_chain)):
            raise ValueError('UQpy: nsamples_per_chain should be a list of integers')
        if self.verbose:
            print('UQpy: Running Bayesian Model Selection.')
        # Perform MCMC for all candidate models
        for i, (inference_model, bayes_estimator) in enumerate(zip(self.candidate_models, self.bayes_estimators)):
            if self.verbose:
                print('UQpy: Running MCMC for model '+inference_model.name)
            if nsamples is not None:
                bayes_estimator.run(nsamples=nsamples[i])
            elif nsamples_per_chain is not None:
                bayes_estimator.run(nsamples_per_chain=nsamples_per_chain[i])
            else:
                raise ValueError('UQpy: ither nsamples or nsamples_per_chain should be non None')
            self.evidences[i] = self._estimate_evidence(
                method_evidence_computation=self.method_evidence_computation,
                inference_model=inference_model, posterior_samples=bayes_estimator.sampler.samples,
                log_posterior_values=bayes_estimator.sampler.log_pdf_values)

        # Compute posterior probabilities
        self.probabilities = self._compute_posterior_probabilities(
            prior_probabilities=self.prior_probabilities, evidence_values=self.evidences)

        if self.verbose:
            print('UQpy: Bayesian Model Selection analysis completed!')

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
    def _estimate_evidence(method_evidence_computation, inference_model, posterior_samples, log_posterior_values):
        """
        Compute the model evidence, given samples from the parameter posterior pdf.

        As of V3, only the harmonic mean method is supported for evidence computation. This function
        is a utility function (static method), called within the run_estimation method.

        **Inputs:**

        :param method_evidence_computation: Method for evidence computation. As of v3, only the harmonic mean is
                                            supported.
        :type method_evidence_computation: str

        :param inference_model: Inference model.
        :type inference_model: object of class InferenceModel

        :param posterior_samples: Samples from parameter posterior density.
        :type posterior_samples: ndarray of shape (nsamples, nparams)

        :param log_posterior_values: Log-posterior values of the posterior samples.
        :type log_posterior_values: ndarray of shape (nsamples, )

        **Output/Returns:**

        :return evidence: Value of evidence p(data|M).
        :rtype evidence: float

        """
        if method_evidence_computation.lower() == 'harmonic_mean':
            # samples[int(0.5 * len(samples)):]
            log_likelihood_values = log_posterior_values - inference_model.prior.log_pdf(x=posterior_samples)
            temp = np.mean(1./np.exp(log_likelihood_values))
        else:
            raise ValueError('UQpy: Only the harmonic mean method is currently supported')
        return 1./temp

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
        scaled_evidences = [evi * prior_prob for (evi, prior_prob) in zip(evidence_values, prior_probabilities)]
        return scaled_evidences / np.sum(scaled_evidences)






# UQpy is distributed under the MIT license.
#
# Copyright (C) 2018  -- Michael D. Shields
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This module contains classes and functions for statistical inference from data. The module currently contains the
following classes:

* InferenceModel: Define a probabilistic model for Inference.
* MLEstimation: Compute maximum likelihood parameter estimate.
* InfoModelSelection: Perform model selection using information theoretic criteria.
* BayesParameterEstimation: Perform Bayesian parameter estimation (estimate posterior density) via MCMC or IS.
* BayesModelSelection: Estimate model posterior probabilities.
"""

from UQpy.RunModel import RunModel
from UQpy.SampleMethods import MCMC, IS
from UQpy.Utilities import check_input_dims
from UQpy.Distributions import Distribution

import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize


########################################################################################################################
########################################################################################################################
#                            Define the model - probability model or python model
########################################################################################################################

class InferenceModel:
    """
    Define a probabilistic model for inference.

    This class defines an inference model that will serve as input for all remaining inference classes. A model can be
    defined in various ways:
    - case 1a: Gaussian error model powered by RunModel, i.e., data ~ h(theta) + eps, where eps is iid Gaussian and h
    consists in running RunModel. Data is a 1D ndarray in this setting.
    - case 1b: non-Gaussian error model powered by RunModel, the user must provide the likelihood function in addition
    to a RunModel object. The data type is user-defined and must be consistent with the likelihood function definition.
    - case 2: the likelihood function is user-defined and does not leverage RunModel. The data type must be consistent
    with the likelihood function definition.
    - case 3: Learn parameters of a probability distribution pi (in dimension dim). Data is an ndarray of shape
    (ndata, dim) and consists in ndata iid samples from pi. The user must define the distribution_object input.

    **Input:**

    :param nparams: Number of parameters to be estimated.

                    This input must be specified.
    :type nparams: int

    :param name: Name of model - optional but useful in a model selection setting.
    :type name: string

    :param run_model_object: RunModel class object that defines the forward model.

                             This input is required for cases 1a and 1b.
    :type run_model_object: object of class RunModel

    :param log_likelihood: Function that defines the log-likelihood model, possibly in conjunction with the
                           run_model_object.

                           Default is None, then a Gaussian-error model is considered (case 1a). It must be provided for
                           cases 1b and 2.
    :type log_likelihood: callable

    :param distribution_object: Distribution pi for which to learn parameters from iid data.

                                This input is required for case 3.
    :type distribution_object: object of Distribution class

    :param error_covariance: Covariance for Gaussian error model (case 1a).

                             Default is 1.
    :type error_covariance: ndarray or float

    :param prior: Prior distribution.
    :type prior: object of Distribution class

    :param prior_params: Parameters of the prior pdf.
    :type prior_params: ndarray or list of ndarrays

    :param prior_copula_params: Parameters of the copula of the prior.
    :param prior_copula_params: ndarray or list of ndarrays

    :param kwargs_likelihood: Additional keyword arguments for the log-likelihood function.
    :type kwargs_likelihood: dictionary

    **Authors:**

    Audrey Olivier

    Last Modified: 12/19 by Audrey Olivier
    """

    def __init__(self, nparams, run_model_object=None, log_likelihood=None, distribution_object=None, name='',
                 error_covariance=1.0, prior=None, prior_params=None, prior_copula_params=None,
                 verbose=False, **kwargs_likelihood
                 ):

        # Initialize some parameters
        self.nparams = nparams
        if not isinstance(self.nparams, int) or self.nparams <= 0:
            raise TypeError('Input nparams must be an integer > 0.')
        self.name = name
        if not isinstance(self.name, str):
            raise TypeError('Input name must be a string.')
        self.verbose = verbose

        # Perform checks on inputs run_model_object, log_likelihood, distribution_object that define the inference model
        if (run_model_object is None) and (log_likelihood is None) and (distribution_object is None):
            raise ValueError('One of run_model_object, log_likelihood or distribution_object inputs must be provided.')
        if run_model_object is not None and (not isinstance(run_model_object, RunModel)):
            raise TypeError('Input run_model_object should be an object of class RunModel.')
        if (log_likelihood is not None) and (not callable(log_likelihood)):
            raise TypeError('Input log_likelihood should be a callable.')
        if distribution_object is not None:
            if (run_model_object is not None) or (log_likelihood is not None):
                raise ValueError('Input distribution_object cannot be provided concurrently with log_likelihood or '
                                 'run_model_object.')
            if not isinstance(distribution_object, Distribution):
                raise TypeError('Input distribution_object should be an object of class Distribution.')
            if not hasattr(distribution_object, 'log_pdf'):
                if not hasattr(distribution_object, 'pdf'):
                    raise AttributeError('distribution_object should have a log_pdf or pdf method')
                distribution_object.log_pdf = lambda x: np.log(distribution_object.pdf(x))
            if self.name == '':
                self.name = distribution_object.dist_name

        self.run_model_object = run_model_object
        self.error_covariance = error_covariance
        self.log_likelihood = log_likelihood
        self.kwargs_likelihood = kwargs_likelihood
        self.distribution_object = distribution_object

        # Define prior if it is given, and set its parameters if provided
        if prior is not None:
            prior.update_params(params=prior_params, copula_params=prior_copula_params)
            if not hasattr(prior, 'log_pdf'):
                if not hasattr(prior, 'pdf'):
                    raise AttributeError('prior should have a log_pdf or pdf method')
                prior.log_pdf = lambda x: np.log(prior.pdf(x))
        self.prior = prior

    def evaluate_log_likelihood(self, params, data):
        """
        Evaluate the log likelihood log p(data|params).

        This method is the central piece for the Inference module, it is being called repeatedly by all other inference
        classes to evaluate the likelihood of the data. The log likelihood can be evaluated at several parameter vectors
        at once, i.e., params is a (nsamples, nparams) ndarray. If the inference model is powered by RunModel the
        RunModel.run method is called here, possibly leveraging its serial/parallel execution.

        **Inputs:**

        :param params: Parameter vector(s) at which to evaluate the likelihood function.
        :type params: ndarray of shape (nsamples, nparams)

        :param data: Data from which to learn.
        :type data: ndarray

        **Output/Returns:**

        :param log_like_values: Log-likelihood evaluated at all nsamples parameter vector values.
        :type log_like_values: ndarray of shape (nsamples, )
        """

        params = check_input_dims(params)
        if params.shape[1] != self.nparams:
            raise ValueError('Wrong dimensions in params.')

        # Case 1 - Forward model is given by RunModel
        if self.run_model_object is not None:
            self.run_model_object.run(samples=params, append_samples=False)
            model_outputs = self.run_model_object.qoi_list

            # Case 1.a: Gaussian error model
            if self.log_likelihood is None:
                log_like_values = np.array(
                    [multivariate_normal.logpdf(data, mean=np.array(outpt).reshape((-1,)), cov=self.error_covariance)
                     for outpt in model_outputs]
                )

            # Case 1.b: likelihood is user-defined
            else:
                log_like_values = self.log_likelihood(
                    data=data, model_outputs=model_outputs, params=params, **self.kwargs_likelihood
                )
                if not isinstance(log_like_values, np.ndarray):
                    log_like_values = np.array(log_like_values)
                if log_like_values.shape != (params.shape[0],):
                    raise ValueError('Likelihood function should output a (nsamples, ) nd array of likelihood values.')

        # Case 2 - Log likelihood is user defined
        elif self.log_likelihood is not None:
            log_like_values = self.log_likelihood(data=data, params=params, **self.kwargs_likelihood)
            if not isinstance(log_like_values, np.ndarray):
                log_like_values = np.array(log_like_values)
            if log_like_values.shape != (params.shape[0],):
                raise ValueError('Likelihood function should output a (nsamples, ) nd array of likelihood values.')

        # Case 3 - Learn parameters of a probability distribution pi. Data consists in iid sampled from pi.
        else:
            log_like_values = np.array([np.sum(self.distribution_object.log_pdf(x=data, params=params_))
                                        for params_ in params])

        return log_like_values

    def evaluate_log_posterior(self, params, data):
        """
        Evaluate the scaled log posterior log [ p(data|params) p(prior) ].

        This method is called by classes that perform Bayesian inference. If the Inference model does not possess a
        prior, an uninformative prior p(params)=1 is assumed.

        **Inputs:**

        :param params: Parameter vector(s) at which to evaluate the log posterior function.
        :type params: ndarray of shape (nsamples, nparams)

        :param data: Data from which to learn.
        :type data: ndarray

        **Output/Returns:**

        :param log_posterior: Log-posterior evaluated at all nsamples parameter vector values.
        :type log_posterior: ndarray of shape (nsamples, )
        """

        # Compute log likelihood
        log_likelihood_eval = self.evaluate_log_likelihood(params=params, data=data)

        # If the prior is not provided it is set to an non-informative prior p(theta)=1, log_posterior = log_likelihood
        if self.prior is None:
            return log_likelihood_eval

        # Otherwise, use prior provided in the InferenceModel setup
        log_prior_eval = self.prior.log_pdf(x=params)

        return log_likelihood_eval + log_prior_eval


########################################################################################################################
########################################################################################################################
#                                  Maximum Likelihood Estimation
########################################################################################################################

class MLEstimation:
    """
    Evaluate the maximum likelihood estimate of a model given some data.

    Perform maximum likelihood estimation, i.e., given some data, compute the parameter vector that maximizes the
    likelihood p(data|theta).

    **Inputs:**

    :param model: The inference model that defines the likelihood function.
    :type model: object of class InferenceModel

    :param data: Available data
    :type data: ndarray or consistent with log likelihood function in InferenceModel

    :param optimizer: Optimization algorithm used to compute the mle.
                      This callable takes in as inputs the function to be minimized and an initial guess and returns
                      an object with attributes x (minimizer) and fun (minimum function value). See scipy.optimize.
                      Default is None, then the optimizer used is scipy.optimize.minimize.
    :type optimizer: callable

    :param x0: Starting point(s) for optimization. If not provided, see iter_optim.
    :type x0: ndarray of shape (n_starts, nparams) or (nparams, )

    :param iter_optim: number of iterations that the optimization is run, starting at random initial guesses. It is only
                       used if x0 is not provided. If neither x0 nor iter_optim are provided, the optimization is not
                       performed (see run_estiamtion method).
    :type iter_optim: integer

    :param kwargs: Additional keyword arguments to the optimizer
    :type kwargs: dictionary

    **Attributes:**

    :param: mle: value of parameter vector that maximizes the likelihood function
    :type: mle: ndarray of shape (nparams, )

    :param: max_log_like: value of the likelihood function at the MLE
    :type: max_log_like: float

    **Authors:**

    Audrey Olivier, Dimitris Giovanis

    Last Modified: 12/19 by Audrey Olivier

    """

    def __init__(self, inference_model, data, verbose=False, iter_optim=None, x0=None, optimizer=None, **kwargs):

        # Initialize variables
        self.inference_model = inference_model
        if not isinstance(inference_model, InferenceModel):
            raise TypeError('Input inference_model should be of type InferenceModel')
        self.data = data
        self.kwargs_optim = kwargs
        self.verbose = verbose
        if optimizer is None:
            self.optimizer = minimize
        elif callable(optimizer):
            self.optimizer = optimizer
        else:
            raise TypeError('Input optimizer should be None or a callable.')
        self.mle = None
        self.max_log_like = None
        if self.verbose:
            print('Initialization of MLEstimation object completed.')

        # Run the optimization procedure
        if (iter_optim is not None) or (x0 is not None):
            self.run_estimation(iter_optim=iter_optim, x0=x0)

    def run_estimation(self, iter_optim=1, x0=None):
        """
        Run the maximum likelihood estimation procedure.

        This function runs the optimization and updates the mle and max_log_like attributes of the class. If the
        parameters of a distribution pi are being learnt (case 3), the fit method of that distribution (it it exists)
        is called to compute the MLE instead of running the optimization. If x0 or iter_optim are given when creating
        the MLEstimation object, this method is called directly when the object is created.

        **Inputs:**

        :param x0: Starting point(s) for optimization. Default is None. If not provided, see iter_optim.
        :type x0: ndarray of shape (n_starts, nparams) or (nparams, )

        :param iter_optim: Number of iterations that the optimization is run, starting at random initial guesses. It is
                           only used if x0 is not provided.

                           Default is 1.
        :type iter_optim: integer
        """

        # Case 3: check if the distribution pi has a fit method, can be used for MLE. If not, use optimization below.
        if (self.inference_model.distribution_object is not None) and \
                hasattr(self.inference_model.distribution_object, 'fit'):
            if not (isinstance(iter_optim, int) and iter_optim >= 1):
                raise ValueError('iter_optim should be an integer >= 1.')
            if self.verbose:
                print('Evaluating maximum likelihood estimate for inference model ' + self.inference_model.name +
                      ', using fit method.')
            for _ in range(iter_optim):
                mle_tmp = np.array(self.inference_model.distribution_object.fit(self.data))
                max_log_like_tmp = self.inference_model.evaluate_log_likelihood(
                    params=mle_tmp[np.newaxis, :], data=self.data)[0]
                # Save result
                if self.mle is None:
                    self.mle = mle_tmp
                    self.max_log_like = max_log_like_tmp
                else:
                    if max_log_like_tmp > self.max_log_like:
                        self.mle = mle_tmp
                        self.max_log_like = max_log_like_tmp

        # Other cases: run optimization (use x0 if provided, otherwise sample starting point from [0, 1] or bounds)
        else:
            if self.verbose:
                print('Evaluating maximum likelihood estimate for inference model ' + self.inference_model.name +
                      ', via optimization.')
            if x0 is None:
                if not (isinstance(iter_optim, int) and iter_optim >= 1):
                    raise ValueError('iter_optim should be an integer >= 1.')
                x0 = np.random.rand(iter_optim, self.inference_model.nparams)
                if 'bounds' in self.kwargs_optim.keys():
                    bounds = np.array(self.kwargs_optim['bounds'])
                    x0 = bounds[:, 0].reshape((1, -1)) + (bounds[:, 1] - bounds[:, 0]).reshape((1, -1)) * x0
            else:
                x0 = np.atleast_2d(x0)
                if x0.shape[1] != self.inference_model.nparams:
                    raise ValueError('Wrong dimensions in x0')
            for x0_ in x0:
                res = self.optimizer(self.evaluate_neg_log_likelihood_data, x0_, **self.kwargs_optim)
                mle_tmp = res.x
                max_log_like_tmp = (-1) * res.fun
                # Save result
                if self.mle is None:
                    self.mle = mle_tmp
                    self.max_log_like = max_log_like_tmp
                else:
                    if max_log_like_tmp > self.max_log_like:
                        self.mle = mle_tmp
                        self.max_log_like = max_log_like_tmp
        if self.verbose:
            print('ML estimation completed.')

    def evaluate_neg_log_likelihood_data(self, one_param):
        """
        Compute negative log likelihood for one parameter vector.

        This is the function to be minimized in the optimization procedure. This is a utility function that will not be
        called by the user.

        **Inputs:**

        :param one_param: A single parameter vector.
        :type one_param: ndarray of shape (nparams, )

        **Output/Returns:**

        :param neg_log_like: negative log-likelihood.
        :type neg_log_like: float
        """

        return -1 * self.inference_model.evaluate_log_likelihood(params=one_param.reshape((1, -1)), data=self.data)[0]


########################################################################################################################
########################################################################################################################
#                                  Model Selection Using Information Theoretic Criteria
########################################################################################################################

class InfoModelSelection:
    """
    Perform model selection using information theoretic criteria.

    Supported criteria are BIC, AIC (default), AICc. This class leverages the MLEstimation class for maximum likelihood
    estimation.

    **Inputs:**

    :param candidate_models: Candidate models, must be a list of objects of class InferenceModel
    :type candidate_models: list

    :param data: Available data
    :type data: ndarray

    :param criterion: Criterion to be used (AIC, BIC, AICc)

                      Default is 'AIC'
    :type criterion: str

    :param x0: starting points for optimization - see MLEstimation
    :type x0: list (length nmodels) of ndarrays

    :param iter_optim: number of iterations for the maximization procedure - see MLEstimation
    :type iter_optim: list (length nmodels) of integers

    :param kwargs: Additional keyword inputs to the maximum likelihood estimator for each model
    :type kwargs: dictionary, each value is a list of length nmodels

    **Attributes:**

    :return ml_estimators: MLEstimation results for all models (contains e.g. fitted parameters).
    :rtype ml_estimators: list (length nmodels) of MLEstimation objects

    :return criterion_values: Value of the criterion for all models.
    :rtype criterion_values: list (length nmodels) of floats

    :return penalty_terms: Value of the penalty term for all models. Data fit term is then
                           criterion_value - penalty_term.
    :rtype penalty_terms: list (length nmodels) of floats

    :return probabilities: Value of the model probabilities, p = exp(-criterion/2).
    :rtype probabilities: list (length nmodels) of floats

    **Authors:**

    Audrey Olivier, Dimitris Giovanis

    Last Modified: 12/19 by Audrey Olivier

    """

    def __init__(self, candidate_models, data, criterion='AIC', verbose=False, iter_optim=None, x0=None, **kwargs):

        # Check inputs
        # candidate_models is a list of InferenceModel objects
        if not isinstance(candidate_models, (list, tuple)) or not all(isinstance(model, InferenceModel)
                                                                      for model in candidate_models):
            raise TypeError('Input candidate_models must be a list of InferenceModel objects.')
        self.nmodels = len(candidate_models)
        self.candidate_models = candidate_models
        self.data = data
        if criterion not in ['AIC', 'BIC', 'AICc']:
            raise ValueError('Criterion should be AIC (default), BIC or AICc')
        self.criterion = criterion
        self.verbose = verbose

        # Instantiate the ML estimators
        if not all(isinstance(value, (list, tuple)) for (key, value) in kwargs.items()) or \
                not all(len(value) == len(candidate_models) for (key, value) in kwargs.items()):
            raise TypeError('Extra inputs to model selection must be lists of length len(candidate_models)')
        self.ml_estimators = []
        for i, inference_model in enumerate(self.candidate_models):
            kwargs_i = dict([(key, value[i]) for (key, value) in kwargs.items()])
            ml_estimator = MLEstimation(inference_model=inference_model, data=self.data, verbose=self.verbose,
                                        x0=None, iter_optim=None, **kwargs_i, )
            self.ml_estimators.append(ml_estimator)

        # Initialize the outputs
        self.criterion_values = [None] * self.nmodels
        self.penalty_terms = [None] * self.nmodels
        self.probabilities = [None] * self.nmodels

        # Run the model selection procedure
        if (iter_optim is not None) or (x0 is not None):
            self.run_estimation(iter_optim=iter_optim, x0=x0)

    def run_estimation(self, iter_optim=1, x0=None):
        """
        Run the model selection procedure, i.e., compute criterion for all models.

        This function calls the run_estimation method of the MLEstimation object for each model to compute the maximum
        log likelihood, then computes the value of the criterion for all models. This function updates attributes
        ml_estimators, criterion_values, penalty_terms and probabilities. If x0 or iter_optim are given when creating
        the object, this method is called directly when the object is created.

        **Inputs:**

        :param x0: Starting point(s) for optimization for all models. Default is None. If not provided, see iter_optim.
        :type x0: list (length nmodels) of ndarrays

        :param iter_optim: number of iterations that the optimization is run, starting at random initial guesses. It is
                           only used if x0 is not provided.

                           Default is 1.
        :type iter_optim: integer or list (length nmodels) of integers

        """

        # Check inputs x0, iter_optim
        if isinstance(iter_optim, int) or iter_optim is None:
            iter_optim = [iter_optim] * self.nmodels
        if not (isinstance(iter_optim, list) and len(iter_optim) == self.nmodels):
            raise ValueError('iter_optim should be an int or list of length nmodels')
        if x0 is None:
            x0 = [None] * self.nmodels
        if not (isinstance(x0, list) and len(x0) == self.nmodels):
            raise ValueError('x0 should be a list of length nmodels (or None).')

        # Loop over all the models
        for i, (inference_model, ml_estimator) in enumerate(zip(self.candidate_models, self.ml_estimators)):
            # First evaluate ML estimate for all models, do several iterations if demanded
            ml_estimator.run_estimation(iter_optim=iter_optim[i], x0=x0[i])

            # Then minimize the criterion
            self.criterion_values[i], self.penalty_terms[i] = self.compute_info_criterion(
                criterion=self.criterion, data=self.data, inference_model=inference_model,
                max_log_like=ml_estimator.max_log_like, return_penalty=True)

        # Compute probabilities from criterion values
        self.probabilities = self.compute_probabilities(self.criterion_values)

    def sort_models(self):
        """
        Sort models in descending order of model probability (increasing order of criterion value).

        This function sorts - in place - the attribute lists candidate_models, ml_estimators, criterion_values,
        penalty_terms and probabilities so that they are sorted from most probable to least probable model. It is a
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
    def compute_info_criterion(criterion, data, inference_model, max_log_like, return_penalty=False):
        """
        Compute the criterion value for a given model, given a max_log_likelihood value.

        The criterion value is -2 * max_log_like + penalty, the penalty depends on the chosen criterion. This function
        is a utility function (static method), called within the run_estimation method.

        **Inputs:**

        :param criterion: Chosen criterion.
        :type criterion: str

        :param data: Available data.
        :type data: ndarray

        :param inference_model: Inference model.
        :type inference_model: object of class InferenceModel

        :param max_log_like: Value of likelihood function at MLE.
        :type max_log_like: float

        :param return_penalty: Boolean that sets whether to return the penalty term as additional output.

                               Default is False
        :type return_penalty: bool

        **Output/Returns:**

        :param criterion_value: Value of criterion.
        :type criterion_value: float

        :param penalty_term: Value of penalty term.
        :type penalty_term: float

        """

        n_params = inference_model.nparams
        ndata = len(data)
        if criterion == 'BIC':
            penalty_term = np.log(ndata) * n_params
        elif criterion == 'AICc':
            penalty_term = 2 * n_params + (2 * n_params ** 2 + 2 * n_params) / (ndata - n_params - 1)
        elif criterion == 'AIC':  # default
            penalty_term = 2 * n_params
        else:
            raise ValueError('Criterion should be AIC (default), BIC or AICc')
        if return_penalty:
            return -2 * max_log_like + penalty_term, penalty_term
        return -2 * max_log_like + penalty_term

    @staticmethod
    def compute_probabilities(criterion_values):
        """
        Compute the model probability given criterion values for all models.

        Model probability is proportional to exp(-criterion/2), model probabilities over all models sum up to 1. This
        function is a utility function (static method), called within the run_estimation method.

        **Inputs:**

        :param criterion_values: Values of criterion for all models.
        :type criterion_values: list (length nmodels) of floats

        **Output/Returns:**

        :param probabilities: Values of model probabilities
        :type probabilities: list (length nmodels) of floats

        """

        delta = np.array(criterion_values) - min(criterion_values)
        prob = np.exp(-delta / 2)
        return prob / np.sum(prob)


########################################################################################################################
########################################################################################################################
#                                  Bayesian Parameter estimation
########################################################################################################################

class BayesParameterEstimation:
    """
    Estimate the parameter posterior density given some data.

    This class generates samples from the parameter posterior distribution, using MCMC or IS. It leverages the MCMC and
    IS classes from the SampleMethods module.

    **Inputs:**

    :param inference_model: Model for inference
    :type inference_model: object of class InferenceModel

    :param data: Available data
    :type data: ndarray

    :param sampling_method: Sampling method to be used, 'MCMC' or 'IS'.

                            Default is 'MCMC'
    :type sampling_method: str

    :param nsamples: Number of samples used in MCMC/IS
    :type nsamples: int

    :param nsamples_per_chain: Number of samples per chain used in MCMC (not used if nsamples is defined)
    :type nsamples_per_chain: int

    :param nchains: Number of chains in MCMC, will be used to sample seed from prior if seed is not provided.

                    Default is 1.
    :type nchains: int

    :param kwargs: Additional keyword inputs to the sampling method, see MCMC and IS
    :type kwargs: dictionary

    **Attributes:**

    :return sampler: sampling method object, contains e.g. the samples
    :rtype sampler: object of class SampleMethods.MCMC or SampleMethods.IS

    **Authors:**

    Audrey Olivier, Dimitris Giovanis

    Last Modified: 12/19 by Audrey Olivier

    """

    def __init__(self, inference_model, data, sampling_method='MCMC', nsamples=None, nsamples_per_chain=None, nchains=1,
                 verbose=False, **kwargs):

        self.inference_model = inference_model
        if not isinstance(self.inference_model, InferenceModel):
            raise TypeError('Input inference_model should be of type InferenceModel')
        self.data = data
        self.sampling_method = sampling_method
        self.verbose = verbose

        if self.sampling_method == 'MCMC':
            # If the seed is not provided, sample one from the prior pdf of the parameters
            if 'seed' not in kwargs.keys() or kwargs['seed'] is None:
                if self.inference_model.prior is None or not hasattr(self.inference_model.prior, 'rvs'):
                    raise NotImplementedError('A prior with a rvs method or a seed must be provided for MCMC.')
                else:
                    kwargs['seed'] = self.inference_model.prior.rvs(nsamples=nchains)
            self.sampler = MCMC(dimension=self.inference_model.nparams, verbose=self.verbose,
                                log_pdf_target=self.inference_model.evaluate_log_posterior, args_target=(self.data, ),
                                **kwargs)

        elif self.sampling_method == 'IS':
            # Importance distribution is either given by the user, or it is set as the prior of the model
            if 'proposal' not in kwargs or kwargs['proposal'] is None:
                if self.inference_model.prior is None:
                    raise NotImplementedError('A proposal density or a prior must be provided.')
                kwargs['proposal'] = self.inference_model.prior

            self.sampler = IS(log_pdf_target=self.inference_model.evaluate_log_posterior, args_target=(self.data, ),
                              verbose=self.verbose, **kwargs)

        else:
            raise ValueError('Sampling_method should be either "MCMC" or "IS"')

        if self.verbose:
            print('Initialization of sampling technique completed successfully!')

        # Run the analysis if a certain number of samples was provided
        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run_estimation(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

    def run_estimation(self, nsamples=None, nsamples_per_chain=None):
        """
        Run the Bayesian inference procedure, i.e., sample from the parameter posterior distribution.

        This function calls the run method of MCMC/IS to generate samples from the parameter posterior distribution. It
        updates the sampler attribute. If nsamples or nsamples_per_chain are given when creating the object, this method
        is called directly when the object is created. It can also be called separately.

        **Inputs:**

        :param nsamples: Number of samples used in MCMC/IS. Either nsamples or nsamples_per_chain must be provided.
        :type nsamples: int

        :param nsamples_per_chain: Number of samples per chain used in MCMC (not used in IS or if nsamples is defined).
        :type nsamples_per_chain: int

        """

        if isinstance(self.sampler, MCMC):
            self.sampler.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)
            #self.samples = self.sampler.samples

        elif isinstance(self.sampler, IS):
            if nsamples_per_chain is not None:
                raise ValueError('nsamples_per_chain is not an appropriate input for IS.')
            self.sampler.run(nsamples=nsamples)
            #self.samples, self.weights = self.sampler.samples, self.sampler.weights

        if self.verbose:
            print('Running parameter estimation with ' + self.sampling_method + ' completed successfully!')


########################################################################################################################
########################################################################################################################
#                                  Bayesian Model Selection
########################################################################################################################


class BayesModelSelection:

    """
    Perform model selection via Bayesian inference, i.e., compute model posterior probabilities given data.

    This class leverages the BayesParameterEstimation class to get samples from the parameter posterior densities. These
    samples are then used to compute the model evidence p(data|model) for all models and the model posterior
    probabilities.

    **References:**

    1. A.E. Raftery, M.A. Newton, J.M. Satagopan, and P.N. Krivitsky. "Estimating the integrated likelihood via
       posterior simulation using the harmonic mean identity". In Bayesian Statistics 8, pages 1–45, 2007.

    **Inputs:**

    :param candidate_models: Candidate models, must be a list of objects of class InferenceModel
    :type candidate_models: list

    :param data: Available data
    :type data: ndarray

    :param prior_probabilities: Prior probabilities of each model, default is 1/nmodels for all models
    :type prior_probabilities: list of floats

    :param method_evidence_computation: as of v3, only the harmonic mean method is supported
    :type method_evidence_computation: str

    :param kwargs: Additional keyword inputs to the BayesParameterEstimation class, for all models
    :type kwargs: dictionary, each value is a list of length nmodels

    :param nsamples: number of samples used in MCMC
    :type nsamples: list (length nmodels) of integers

    :param nsamples_per_chain: number of samples per chain used in MCMC (not used if nsamples is defined)
    :type nsamples_per_chain: list (length nmodels) of integers

    **Attributes:**

    :return bayes_estimators: results of the Bayesian parameter estimation
    :rtype bayes_estimators: list (length nmodels) of BayesParameterEstimation objects

    :return evidences: value of the evidence for all models
    :rtype evidences: list (length nmodels) of floats

    :return probabilities: posterior probability for all models
    :rtype probabilities: list (length nmodels) of floats

    **Authors:**

    Audrey Olivier, Yuchen Zhou

    Last Modified: 01/24/2020 by Audrey Olivier

    """

    def __init__(self, candidate_models, data, prior_probabilities=None, method_evidence_computation='harmonic_mean',
                 verbose=False, nsamples=None, nsamples_per_chain=None, **kwargs):

        # Check inputs: candidate_models is a list of instances of Model, data must be provided, and input arguments
        # for MCMC must be provided as a list of length len(candidate_models)
        if (not isinstance(candidate_models, list)) or (not all(isinstance(model, InferenceModel)
                                                                for model in candidate_models)):
            raise TypeError('A list InferenceModel objects must be provided.')
        self.candidate_models = candidate_models
        self.nmodels = len(candidate_models)
        self.data = data
        self.method_evidence_computation = method_evidence_computation
        self.verbose = verbose

        if prior_probabilities is None:
            self.prior_probabilities = [1. / len(candidate_models) for _ in candidate_models]
        else:
            self.prior_probabilities = prior_probabilities

        # Instantiate the Bayesian parameter estimators (without running them)
        self.bayes_estimators = []
        if not all(isinstance(value, (list, tuple)) for (key, value) in kwargs.items()) or not all(
                len(value) == len(candidate_models) for (key, value) in kwargs.items()):
            raise TypeError('Extra inputs to model selection must be lists of length len(candidate_models)')
        for i, inference_model in enumerate(self.candidate_models):
            kwargs_i = dict([(key, value[i]) for (key, value) in kwargs.items()])
            kwargs_i.update({'concat_chains_': True, 'save_log_pdf': True})
            bayes_estimator = BayesParameterEstimation(
                inference_model=inference_model, data=self.data, verbose=self.verbose, sampling_method='MCMC',
                nsamples=None, nsamples_per_chain=None, **kwargs_i)
            self.bayes_estimators.append(bayes_estimator)

        # Initialize the outputs
        self.evidences = [0.] * self.nmodels
        self.probabilities = [0.] * self.nmodels

        # Run the model selection procedure
        if nsamples is not None or nsamples_per_chain is not None:
            self.run_estimation(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

    def run_estimation(self, nsamples=None, nsamples_per_chain=None):

        """
        Run the Bayesian model selection procedure, i.e., compute model posterior probabilities.

        This function calls the run_estimation method of the BayesParameterEstimation object for each model to sample
        from the parameter posterior probability, then computes the model evidence and model posterior probability.
        This function updates attributes bayes_estimators, evidences and probabilities. If nsamples or
        nsamples_per_chain are given when creating the object, this method is called directly when the object is
        created. It can also be called separately.

        **Inputs:**

        :param nsamples: number of samples used in MCMC
        :type nsamples: list (length nmodels) of integers

        :param nsamples_per_chain: number of samples per chain used in MCMC (not used if nsamples is defined)
        :type nsamples_per_chain: list (length nmodels) of integers

        """

        if nsamples is not None and not (isinstance(nsamples, list) and len(nsamples) == self.nmodels
                                         and all(isinstance(n, int) for n in nsamples)):
            raise ValueError('nsamples should be a list of integers')
        if nsamples_per_chain is not None and not (isinstance(nsamples_per_chain, list)
                                                   and len(nsamples_per_chain) == self.nmodels
                                                   and all(isinstance(n, int) for n in nsamples_per_chain)):
            raise ValueError('nsamples_per_chain should be a list of integers')
        if self.verbose:
            print('Running Bayesian Model Selection.')
        # Perform MCMC for all candidate models
        for i, (inference_model, bayes_estimator) in enumerate(zip(self.candidate_models, self.bayes_estimators)):
            if self.verbose:
                print('UQpy: Running MCMC for model '+inference_model.name)
            if nsamples is not None:
                bayes_estimator.run_estimation(nsamples=nsamples[i])
            elif nsamples_per_chain is not None:
                bayes_estimator.run_estimation(nsamples_per_chain=nsamples_per_chain[i])
            else:
                raise ValueError('Either nsamples or nsamples_per_chain should be non None')
            self.evidences[i] = self.estimate_evidence(
                method_evidence_computation=self.method_evidence_computation,
                inference_model=inference_model, posterior_samples=bayes_estimator.sampler.samples,
                log_posterior_values=bayes_estimator.sampler.log_pdf_values)

        # Compute posterior probabilities
        self.probabilities = self.compute_posterior_probabilities(
            prior_probabilities=self.prior_probabilities, evidence_values=self.evidences)

        if self.verbose:
            print('Bayesian Model Selection analysis completed!')

    def sort_models(self):
        """
        Sort models in descending order of model probability (increasing order of criterion value).

        This function sorts - in place - the attribute lists candidate_models, prior_probabilities, probabilities and
        evidences so that they are sorted from most probable to least probable model. It is a stand-alone function that
        is provided to help the user to easily visualize which model is the best.

        No inputs/outputs.

        """
        sort_idx = list(np.argsort(np.array(self.probabilities)))[::-1]

        self.candidate_models = [self.candidate_models[i] for i in sort_idx]
        self.prior_probabilities = [self.prior_probabilities[i] for i in sort_idx]
        self.probabilities = [self.probabilities[i] for i in sort_idx]
        self.evidences = [self.evidences[i] for i in sort_idx]

    @staticmethod
    def estimate_evidence(method_evidence_computation, inference_model, posterior_samples, log_posterior_values):
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

        :param evidence: Value of evidence p(data|M).
        :type evidence: float

        """
        if method_evidence_computation.lower() == 'harmonic_mean':
            #samples[int(0.5 * len(samples)):]
            log_likelihood_values = log_posterior_values - inference_model.prior.log_pdf(x=posterior_samples)
            temp = np.mean(1./np.exp(log_likelihood_values))
        else:
            raise ValueError('Only the harmonic mean method is currently supported')
        return 1./temp

    @staticmethod
    def compute_posterior_probabilities(prior_probabilities, evidence_values):
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

        :param probabilities: Values of model posterior probabilities
        :type probabilities: list (length nmodels) of floats

        """
        scaled_evidences = [evi * prior_prob for (evi, prior_prob) in zip(evidence_values, prior_probabilities)]
        return scaled_evidences / np.sum(scaled_evidences)

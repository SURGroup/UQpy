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

"""This module contains functionality for all the Inference supported in UQpy."""

from .RunModel import RunModel
from .SampleMethods import MCMC, IS
from .Utilities import check_input_dims
from .Distributions import Distribution

import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize


########################################################################################################################
########################################################################################################################
#                            Define the model - probability model or python model
########################################################################################################################

class InferenceModel:
    def __init__(self, nparams, run_model_object=None, log_likelihood=None, distribution_object=None, name='',
                 error_covariance=1.0, prior=None, prior_params=None, prior_copula_params=None,
                 verbose=False, **kwargs_likelihood
                 ):

        """
        Define the inference_model. This class possesses two method that, given some data and parameter values,
         evaluate the log likelihood and scaled log posterior (log_likelihood + log_prior)

        Inputs:
            :param nparams: number of parameters to be estimated, required
            :type nparams: int

            :param name: name of model
            :type name: string

            :param run_model_object: RunModel class object that defines the forward model h(param)
            :type run_model_object: object of class RunModel or None

            :param log_likelihood: log-likelihood function. Default is None. This function defines the
            log-likelihood model, possibly in conjunction with the run_model_object.
            :type log_likelihood: callable or None

            :param distribution_object: distribution pi for which to learn parameters from iid data
            :type distribution_object: object of Distribution class

            :param error_covariance: covariance of the Gaussian error for model defined by a python script
            :type error_covariance: ndarray (full covariance matrix) or float (diagonal values)

            :param prior: prior distribution
            :type prior: object of Distribution class

            :param prior_params: parameters of the prior pdf
            :type prior_params: ndarray or list of ndarrays

            :param prior_copula_params: parameters of the copula of the prior, if necessary
            :param prior_copula_params: str

            :param kwargs_likelihood: all additional keyword arguments for the log-likelihood function, if necessary
            :type kwargs_likelihood: dictionary

        """
        self.nparams = nparams
        if not isinstance(self.nparams, int) or self.nparams <= 0:
            raise TypeError('Input nparams must be an integer > 0.')
        self.name = name
        if not isinstance(self.name, str):
            raise TypeError('Input name must be a string.')
        self.verbose = verbose

        # Perform checks on inputs run_model_object, log_likelihood, distribution_object
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
        """ Computes the log-likelihood of model
            inputs: data
                    params, ndarray of dimension (nsamples, nparams)
            output: ndarray of size (nsamples, ), contains log likelihood of p(data | params[i,:]), i=1:nsamples
        """
        params = check_input_dims(params)
        if params.shape[1] != self.nparams:
            raise ValueError('Wrong dimensions in params.')

        # Case 1 - Forward model is given by RunModel
        if self.run_model_object is not None:
            self.run_model_object.run(samples=params, append_samples=False)
            model_outputs = self.run_model_object.qoi_list    # [-params.shape[0]:]

            # Case 1.a: Gaussian error model
            if self.log_likelihood is None:
                results = np.array([multivariate_normal.logpdf(data, mean=np.array(outpt).reshape((-1,)),
                                                               cov=self.error_covariance) for outpt in model_outputs])

            # Case 1.b: likelihood is user-defined
            else:
                results = self.log_likelihood(data=data, model_outputs=model_outputs, params=params,
                                              **self.kwargs_likelihood)
                if not isinstance(results, np.ndarray):
                    results = np.array(results)
                if results.shape != (params.shape[0],):
                    raise ValueError('Likelihood function should output a (nsamples, ) nd array of likelihood values.')

        # Case 2 - Log likelihood is user defined
        elif self.log_likelihood is not None:
            results = self.log_likelihood(data=data, params=params, **self.kwargs_likelihood)
            if not isinstance(results, np.ndarray):
                results = np.array(results)
            if results.shape != (params.shape[0],):
                raise ValueError('Likelihood function should output a (nsamples, ) nd array of likelihood values.')

        # Case 3 - Learn parameters of a probability distribution pi. Data consists in iid sampled from pi.
        else:
            results = np.array([np.sum(self.distribution_object.log_pdf(x=data, params=params_))
                                for params_ in params])

        return results

    def evaluate_log_posterior(self, params, data):
        """ Computes the (scaled) log posterior: log[p(data|params) * p(params)]
            If the Inference model does not possess a prior, an uninformatic prior p(params)=1 is assumed
            inputs: data
                    params, ndarray of dimension (nsamples, nparams)
            output: ndarray of size (nsamples, )
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

    def __init__(self, inference_model, data, verbose=False, iter_optim=None, x0=None, optimizer=None, **kwargs):

        """
        Perform maximum likelihood estimation, i.e., given some data y, compute the parameter vector that maximizes the
        likelihood p(y|theta).

        Inputs:
            :param model: the inference model
            :type model: instance of class InferenceModel

            :param data: Available data
            :type data: ndarray of size (ndata, ) for case 1a or (ndata, dimension) for case 2a, or consistent with
            definition of log_likelihood in the inference_model

            :param optimizer: optimization algorithm used to compute the mle
            :type optimizer: function that takes as first input the function to be minimized, as second input a starting
            point for the optimization. It should return an object with attributes x and fun (the minimizer and the
            value of the function at its minimum). If None (default), the optimizer used is scipy.optimize.minimize

            :param iter_optim: number of iterations for the maximization procedure (each iteration starts at a random
            point)
            :type iter_optim: an integer, default None

            :param x0: starting points for optimization
            :type x0: 1D array (dimension, ) or 2D array (n_starts, dimension), default None

            :param kwargs: input arguments to the optimizer
            :type kwargs: dictionary

        Output:
            :return: mle: value of parameter vector that maximizes the likelihood
            :rtype: mle: ndarray (nparams, )

            :return: max_log_like: value of the maximum likelihood
            :rtype: max_log_like: float

        """
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
        Run optimization
        :param iter_optim: number of iterations of the optimization procedure
        :param x0: starting points for optimization, takes precedence over iter_optim
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
        Compute the negative log-likelihood for one value of the parameter vector (function to be minimized)
        """
        return -1 * self.inference_model.evaluate_log_likelihood(params=one_param.reshape((1, -1)), data=self.data)[0]


########################################################################################################################
########################################################################################################################
#                                  Model Selection Using Information Theoretic Criteria
########################################################################################################################

class InfoModelSelection:

    def __init__(self, candidate_models, data, criterion='AIC', verbose=False, iter_optim=None, x0=None, **kwargs):

        """
            Perform model selection using information theoretic criteria.
            Supported criteria are BIC, AIC (default), AICc.

            Inputs:

            :param candidate_models: Candidate models, must be a list of instances of class InferenceModel
            :type candidate_models: list

            :param data: Available data
            :type data: ndarray

            :param criterion: Criterion to be used (AIC, BIC, AICc)
            :type criterion: str

            :param iter_optim: number of iterations for the maximization procedure - see MLEstimation
            :type iter_optim: list (length nmodels) of integers, default None

            :param x0: starting points for optimization - see MLEstimation
            :type x0: list (length nmodels) of 1D arrays (dimension, ) or 2D arrays (n_starts, dimension), default None

            :param kwargs: inputs to the maximum likelihood estimator, for each model
            :type kwargs: dictionary, each value should be a list of length nmodels

            Outputs:

            :return ml_estimators: MLEstimation results for all models (contains e.g. fitted parameters)
            :rtype ml_estimators: list (length nmodels) of MLEstimation objects

            :return criterion_values: Value of the criterion for all models
            :rtype criterion_values: list (length nmodels) of floats

            :return penalty_terms: Value of the penalty term for all models. Data fit is then
            criterion_value - penalty_term
            :rtype penalty_terms: list (length nmodels) of floats

            :return probabilities: Value of the model probabilities, p = exp(-criterion/2)
            :rtype probabilities: list (length nmodels) of floats

        """

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
        Run estimation, i.e. compute the maximum log likelihood for all models then compute criterion
        :param iter_optim: number of iterations of the optimization procedure
        :param x0: starting points for optimization, takes precedence over iter_optim
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
        Sort models (all outputs lists) in descending order of model probability (increasing order of criterion value)
        """
        sort_idx = list(np.argsort(np.array(self.criterion_values)))

        self.candidate_models = [self.candidate_models[i] for i in sort_idx]
        self.ml_estimators = [self.ml_estimators[i] for i in sort_idx]
        self.criterion_values = [self.criterion_values[i] for i in sort_idx]
        self.penalty_terms = [self.penalty_terms[i] for i in sort_idx]
        self.probabilities = [self.probabilities[i] for i in sort_idx]

    @staticmethod
    def compute_info_criterion(criterion, data, inference_model, max_log_like, return_penalty=False):
        """ Helper function: compute the criterion value, also returns the penalty term if asked for it """
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
        return -2 * max_log_like

    @staticmethod
    def compute_probabilities(criterion_values):
        """ Helper function: compute probability, proportional to exp(-criterion/2) """
        delta = np.array(criterion_values) - min(criterion_values)
        prob = np.exp(-delta / 2)
        return prob / np.sum(prob)


########################################################################################################################
########################################################################################################################
#                                  Inference Parameter estimation
########################################################################################################################

class BayesParameterEstimation:
    """
    Generates samples from the posterior distribution, using MCMC or IS.

    Inputs:

    :param inference_model: model, must be an instance of class InferenceModel
    :type inference_model: list

    :param data: Available data
    :type data: ndarray

    :param sampling_method: Method to be used
    :type sampling_method: str, 'MCMC' or 'IS'

    :param nsamples: number of samples used in MCMC/IS
    :type nsamples: int

    :param nsamples_per_chain: number of samples per chain used in MCMC (not used if nsamples is defined)
    :type nsamples_per_chain: int

    :param nchains: number of chains in MCMC, will be used to sample seed from prior if seed is not provided
    :type nchains: int

    :param kwargs: inputs to the sampling method, see MCMC and IS
    :type kwargs: dictionary

    Outputs:

    :return sampler: sampling object, contains e.g. the samples
    :rtype sampler: SampleMethods.MCMC or SampleMethods.IS object

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
        Run estimation, i.e. generate samples from the posterior (call the run method of MCMC/IS)
        :param nsamples: see MCMC, IS
        :param nsamples_per_chain: see MCMC
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
#                                  Inference Model Selection
########################################################################################################################


class BayesModelSelection:

    def __init__(self, candidate_models, data, prior_probabilities=None, method_evidence_computation='harmonic_mean',
                 verbose=False, nsamples=None, nsamples_per_chain=None, **kwargs):

        """
            Perform model selection using Inference criteria.

            Inputs:

            :param candidate_models: candidate models, must be a list of instances of class InferenceModel
            :type candidate_models: list

            :param data: available data
            :type data: ndarray

            :param prior_probabilities: prior probabilities of each model, default is 1/nmodels for all models
            :type prior_probabilities: list of floats

            :param method_evidence_computation: for now only the harmonic mean is supported
            :type method_evidence_computation: str

            :param kwargs: inputs to the Bayes parameter estimator, for each model
            :type kwargs: dictionary, each value should be a list of length len(candidate_models)

            :param nsamples: number of samples used in MCMC
            :type nsamples: list (length nmodels) of integers

            :param nsamples_per_chain: number of samples per chain used in MCMC (not used if nsamples is defined)
            :type nsamples_per_chain: list (length nmodels) of integers

            Outputs:

            :return bayes_estimators: results of the Bayesian parameter estimation
            :rtype bayes_estimators: list (length nmodels) of BayesParameterEstimation objects

            :return evidence_values: value of the evidence for all models
            :rtype evidence_values: list (length nmodels) of floats

            :return probabilities: posterior probability for all models
            :rtype probabilities: list (length nmodels) of floats

            # Authors: Yuchen Zhou
            # Updated: 01/24/2020 by Audrey Olivier

        """

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
        Sort models (all outputs lists) in descending order of model probability (increasing order of criterion value)
        """
        sort_idx = list(np.argsort(np.array(self.probabilities)))[::-1]

        self.candidate_models = [self.candidate_models[i] for i in sort_idx]
        self.probabilities = [self.probabilities[i] for i in sort_idx]
        self.evidences = [self.evidences[i] for i in sort_idx]

    @staticmethod
    def estimate_evidence(method_evidence_computation, inference_model, posterior_samples, log_posterior_values):
        """ Estimate evidence from MCMC samples, for one model """
        if method_evidence_computation.lower() == 'harmonic_mean':
            #samples[int(0.5 * len(samples)):]
            log_likelihood_values = log_posterior_values - inference_model.prior.log_pdf(x=posterior_samples)
            temp = np.mean(1./np.exp(log_likelihood_values))
        else:
            raise ValueError('Only the harmonic mean method is currently supported')
        return 1./temp

    @staticmethod
    def compute_posterior_probabilities(prior_probabilities, evidence_values):
        """ Compute the posterior probabilities, knowing the values of the evidence """
        scaled_evidences = [evi * prior_prob for (evi, prior_prob) in zip(evidence_values, prior_probabilities)]
        return scaled_evidences / np.sum(scaled_evidences)

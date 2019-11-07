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


from UQpy.SampleMethods import *
from scipy.stats import multivariate_normal
from UQpy.RunModel import RunModel
from scipy.optimize import minimize
import warnings
from .Utilities import check_input_dims
warnings.filterwarnings("ignore")


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
        Define the model, either as a python script, or a probability model

        Inputs:
            :param nparams: number of parameters to be estimated, required
            :type nparams: int

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

            :param kwargs: all additional keyword arguments for the log-likelihood function, if necessary
            :type kwargs: dictionary

        Output:

            :return Model.evaluate_log_likelihood: function that computes the log likelihood at various parameter
            values, given some data
            :rtype Model.evaluate_log_likelihood: callable

            :return Model.evaluate_log_posterior: function that computes the scaled log posterior (log likelihood +
            log prior) at various parameter values, given some data
            :rtype Model.evaluate_log_posterior: callable

            :return Model.prior: probability distribution of the prior
            :rtype Model.prior: instance from Distribution class

            :return Model.kwargs_prior: arguments necessary to call the log_pdf, pdf or rvs method of the prior
            :rtype Model.kwargs_prior: dictionary

        """
        self.nparams = nparams
        if not isinstance(self.nparams, int) or self.nparams <= 0:
            raise TypeError('Input nparams must be an integer > 0.')
        self.name = name
        if not isinstance(self.name, str):
            raise TypeError('Input name must be a string.')
        self.verbose = verbose

        if (run_model_object is None) and (log_likelihood is None) and (distribution_object is None):
            raise ValueError('One of run_model_object, log_likelihood or distribution_object inputs must be provided.')
        if (run_model_object is not None) and (not isinstance(run_model_object, RunModel)):
            raise TypeError('Input run_model_object should be an object of class RunModel.')
        if (log_likelihood is not None) and (not callable(log_likelihood)):
            raise TypeError('Input log_likelihood should be a callable.')
        if (distribution_object is not None) and (not isinstance(distribution_object, Distribution)):
            raise TypeError('Input distribution_object should be an object of class Distribution.')
        if (distribution_object is not None) and ((run_model_object is not None) or (log_likelihood is not None)):
            raise ValueError('Input distribution_object cannot be provided concurrently with log_likelihood or '
                             'run_model_object.')
        self.run_model_object = run_model_object
        self.log_likelihood = log_likelihood
        self.distribution_object = distribution_object
        self.kwargs_likelihood = kwargs_likelihood
        self.error_covariance = error_covariance

        # Define prior if it is given, and set its parameters if provided
        self.prior = prior
        if self.prior is not None:
            self.prior.update_params(params=prior_params, copula_params=prior_copula_params)

    def evaluate_log_likelihood(self, data, params):
        """ Computes the log-likelihood of model
            inputs: data, ndarray of dimension (ndata, )
                    params, ndarray of dimension (nsamples, nparams) or (nparams,)
            output: ndarray of size (nsamples, ), contains log likelihood of p(data | params[i,:])
        """
        params = check_input_dims(params)
        if params.shape[1] != self.nparams:
            raise ValueError('Wrong dimensions in params.')

        # Case 1 - Forward model is given by RunModel
        if self.run_model_object is not None:
            self.run_model_object.run(samples=params, append_samples=False)
            model_outputs = self.run_model_object.qoi_list[-params.shape[0]:]
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
            try:
                results = np.array([np.sum(self.distribution_object.log_pdf(x=data, params=params_))
                                    for params_ in params])
            except AttributeError:
                try:
                    results = np.array([np.sum(np.log(self.distribution_object.pdf(x=data, params=params_)))
                                        for params_ in params])
                except AttributeError:
                    raise AttributeError('Input distribution object should possess a log pdf or a pdf method.')

        return results

    def evaluate_log_posterior(self, data, params):
        """ Computes the log posterior (scaled): log[p(data|params) * p(params)]
            inputs: data, ndarray
                    params, ndarray of dimension (nsamples, nparams) or (nparams,)
            output: ndarray of size (nsamples, ), contains log likelihood of p(data | params[i,:])
        Note: if the Inference model does not possess a prior, an uninformatic prior p(params)=1 is assumed
        """

        # Compute log likelihood
        log_likelihood_eval = self.evaluate_log_likelihood(data=data, params=params)

        # If the prior is not provided it is set to an non-informative prior p(theta)=1, log_posterior = log_likelihood
        if self.prior is None:
            return log_likelihood_eval

        # Otherwise, use prior provided in the InferenceModel setup
        try:
            log_prior_eval = self.prior.log_pdf(x=params)
        except AttributeError:
            try:
                log_prior_eval = np.log(self.prior.pdf(x=params))
            except AttributeError:
                raise AttributeError('The prior of InferenceModel must have a log_pdf or pdf method.')
        return log_likelihood_eval + log_prior_eval


########################################################################################################################
########################################################################################################################
#                                  Maximum Likelihood Estimation
########################################################################################################################

class MLEstimation:

    def __init__(self, inference_model, data, verbose=False, iter_optim=1, x0=None, **kwargs):

        """
        Perform maximum likelihood estimation, i.e., given some data y, compute the parameter vector that maximizes the
        likelihood p(y|theta).

        Inputs:
            :param model: the inference model
            :type model: instance of class InferenceModel

            :param data: Available data
            :type data: ndarray of size (ndata, ) for case 1a or (ndata, dimension) for case 2a, or consistent with
            definition of log_likelihood in the inference_model

            :param iter_optim: number of iterations for the maximization procedure
                (each iteration starts at a random point)
            :type iter_optim: an integer, default 1

            :param x0: starting points for optimization
            :type x0: 1D array (dimension, ) or 2D array (n_starts, dimension)

            :param kwargs: input arguments to scipy.optimize.minimize
            :type kwargs: dictionary

        Output:
            :return: MLEstimation.mle: value of parameter vector that maximizes the likelihood
            :rtype: MLEstimation.mle: ndarray (nparams, )

            :return: MLEstimation.max_log_like: value of the maximum likelihood
            :rtype: MLEstimation.max_log_like: float

        """
        self.inference_model = inference_model
        if not isinstance(inference_model, InferenceModel):
            raise TypeError('Input inference_model should be of type InferenceModel')
        self.data = data
        self.kwargs_optim = kwargs
        self.verbose = verbose

        self.mle = None
        self.max_log_like = None

        # Run the optimization
        if x0 is not None:
            x0 = np.atleast_2d(x0)
            if x0.shape[1] != self.inference_model.nparams:
                raise ValueError('Wrong dimensions in x0')
            iter_optim = x0.shape[0]
        else:
            if not isinstance(iter_optim, int) or iter_optim < 0:
                raise TypeError('Input iter_optim should be a positive integer.')
            x0 = [None] * iter_optim
        if iter_optim >= 1:
            for i in range(iter_optim):
                self.run_estimation(x0=x0[i])

    def run_estimation(self, x0=None):
        """ Run optimization, starting at point x0 (or at a random value if x0 is None). """

        # Case 2: check if the distribution pi has a fit method, can be used for MLE. If not, use optimization.
        if (self.inference_model.distribution_object is not None) and \
                hasattr(self.inference_model.distribution_object, 'fit'):
            if self.verbose:
                print('Evaluating maximum likelihood estimate for inference model ' + self.inference_model.name +
                      ', using fit method.')
            mle_tmp = self.inference_model.distribution_object.fit(self.data)
            max_log_like_tmp = self.inference_model.evaluate_log_likelihood(
                data=self.data, params=mle_tmp[np.newaxis, :])[0]

        # Other cases: just run optimization: use x0 if provided, otherwise sample from [0, 1] or bounds
        else:
            if self.verbose:
                print('Evaluating maximum likelihood estimate for inference model ' + self.inference_model.name +
                      ', via optimization.')
            if x0 is None:
                x0 = np.random.rand(1, self.inference_model.nparams)
                if 'bounds' in self.kwargs_optim.keys():
                    bounds = np.array(self.kwargs_optim['bounds'])
                    x0 = bounds[:, 0].reshape((1, -1)) + (bounds[:, 1] - bounds[:, 0]).reshape((1, -1)) * x0
            else:
                x0 = x0.reshape((1, self.inference_model.nparams))
            res = minimize(self.evaluate_neg_log_likelihood_data, x0, **self.kwargs_optim)
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
            print('Optimization completed.')
        return None

    def evaluate_neg_log_likelihood_data(self, one_param):
        return -1 * self.inference_model.evaluate_log_likelihood(data=self.data, params=one_param.reshape((1, -1)))[0]


########################################################################################################################
########################################################################################################################
#                                  Model Selection Using Information Theoretic Criteria
########################################################################################################################

class InfoModelSelection:

    def __init__(self, candidate_models, data, criterion='AIC', verbose=False, sorted_outputs=True,
                 iter_optim=1, x0=None, **kwargs):

        """
            Perform model selection using information theoretic criteria. Supported criteria are BIC, AIC (default), AICc.

            Inputs:

            :param candidate_models: Candidate models, must be a list of instances of class InferenceModel
            :type candidate_models: list

            :param data: Available data
            :type data: ndarray

            :param criterion: Criterion to be used (AIC, BIC, AICc)
            :type criterion: str

            :param kwargs: inputs to the maximum likelihood estimator, for each model
            :type kwargs: dictionary, each value should be a list of length len(candidate_models)

            :param sorted_outputs: indicates if results are returned in sorted order, according to model
             probabilities
            :type sorted_outputs: bool

            Outputs:

            A list of (sorted) models, their probability based on data and the given criterion, and the parameters
            that maximize the log likelihood. The penalty term (Ockam razor) is also given.

        """

        # Check inputs: candidate_models is a list of instances of Model, data must be provided, and input arguments
        # for ML estimation must be provided as a list of length len(candidat_models)
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
        self.sorted_outputs = sorted_outputs

        # Instantiate the ML estimators
        if not all(isinstance(value, (list, tuple)) for (key, value) in kwargs.items()) or not all(
            len(value) == len(candidate_models) for (key, value) in kwargs.items()):
            raise TypeError('Extra inputs to model selection must be lists of length len(candidate_models)')
        self.ml_estimators = []
        for i, inference_model in enumerate(self.candidate_models):
            kwargs_i = dict([(key, value[i]) for (key, value) in kwargs.items()])
            ml_estimator = MLEstimation(inference_model=inference_model, data=self.data, verbose=self.verbose,
                                        x0=None, iter_optim=0, **kwargs_i, )
            self.ml_estimators.append(ml_estimator)

        # Initialize the outputs
        self.criterion_values = [0.] * self.nmodels
        self.penalty_terms = [0.] * self.nmodels
        self.probabilities = [0.] * self.nmodels

        # Run the model selection procedure
        if iter_optim >= 1 or x0 is not None:
            self.run_estimation(iter_optim=iter_optim, x0=x0)

    def run_estimation(self, iter_optim=1, x0=None):
        # check x0, iter_optim
        if x0 is not None:
            if not (isinstance(x0, list) and len(x0) == self.nmodels):
                raise ValueError('x0 should be a list of length nmodels')
            x0 = [np.atleast_2d(x0_) for x0_ in x0]
            if any(x0_.shape[1] != m.nparams for (x0_, m) in zip(x0, self.candidate_models)):
                raise ValueError('Wrong dimensions in x0')
        else:
            if not isinstance(iter_optim, int):
                raise ValueError('iter_optim should be an integer')
            x0 = [[None] * iter_optim] * self.nmodels

        # Loop over all the models
        for i, (inference_model, ml_estimator) in enumerate(zip(self.candidate_models, self.ml_estimators)):
            # First evaluate ML estimate for all models, do several iterations if demanded
            for x0_ in x0[i]:
                ml_estimator.run_estimation(x0=x0_)

            # Then minimize the criterion
            self.criterion_values[i], self.penalty_terms[i] = self.compute_criterion(
                inference_model=inference_model, max_log_like=ml_estimator.max_log_like, return_penalty=True)

        # Compute probabilities from criterion values
        self.probabilities = self.compute_probabilities(self.criterion_values)
        # Return outputs in sorted order, from most probable model to least probable model
        if self.sorted_outputs:
            sort_idx = list(np.argsort(np.array(self.criterion_values)))

            self.candidate_models = [self.candidate_models[i] for i in sort_idx]
            self.ml_estimators = [self.ml_estimators[i] for i in sort_idx]
            self.criterion_values = [self.criterion_values[i] for i in sort_idx]
            self.penalty_terms = [self.penalty_terms[i] for i in sort_idx]
            self.probabilities = [self.probabilities[i] for i in sort_idx]

    def compute_criterion(self, inference_model, max_log_like, return_penalty=False):
        """ Compute the criterion value, also returns the penalty term if asked for it """
        n_params = inference_model.nparams
        ndata = len(self.data)
        if self.criterion == 'BIC':
            penalty_term = np.log(ndata) * n_params
        elif self.criterion == 'AICc':
            penalty_term = 2 * n_params + (2 * n_params ** 2 + 2 * n_params) / (ndata - n_params - 1)
        elif self.criterion == 'AIC':  # default
            penalty_term = 2 * n_params
        else:
            raise ValueError('Criterion should be AIC (default), BIC or AICc')
        if return_penalty:
            return -2 * max_log_like + penalty_term, penalty_term
        return -2 * max_log_like

    @staticmethod
    def compute_probabilities(criterion_values):
        """ Compute probability, proportional to exp(-criterion/2) """
        delta = np.array(criterion_values) - min(criterion_values)
        prob = np.exp(-delta / 2)
        return prob / np.sum(prob)


########################################################################################################################
########################################################################################################################
#                                  Inference Parameter estimation
########################################################################################################################

class BayesParameterEstimation:

    def __init__(self, inference_model, data, sampling_method='MCMC', nsamples=None, nsamples_per_chain=None, nchains=1,
                 verbose=False, **kwargs):

        """
            Generates samples from the posterior distribution, using MCMC or IS.

            Inputs:

            :param inference_model: model, must be an instance of class InferenceModel
            :type inference_model: list

            :param data: Available data
            :type data: ndarray

            :param sampling_method: Method to be used
            :type sampling_method: str, 'MCMC' or 'IS'

            :param kwargs: inputs to the sampling method, see MCMC and IS
            :type kwargs: dictionary

            Outputs:

            Attributes of bayes = BayesParameterEstimation(...). For MCMC, bayes.samples are samples from the posterior
             pdf. For IS, bayes.samples in combination with bayes.weights provide an estimate of the posterior.

        """

        self.inference_model = inference_model
        if not isinstance(self.inference_model, InferenceModel):
            raise TypeError('Input inference_model should be of type InferenceModel')
        self.data = data
        self.sampling_method = sampling_method
        if not 'nsamples' in kwargs.keys():
            raise ValueError('Input nsamples must be provided.')
        self.verbose = verbose

        if self.sampling_method == 'MCMC':
            # If the seed is not provided, sample one from the prior pdf of the parameters
            if 'seed' not in kwargs.keys() or kwargs['seed'] is None:
                if self.inference_model.prior is None or not hasattr(self.inference_model.prior, 'rvs'):
                    raise NotImplementedError('A prior with a rvs method or a seed must be provided for MCMC.')
                else:
                    kwargs['seed'] = self.inference_model.prior.rvs(nsamples=nchains)
            self.sampler = MCMC(dimension=self.inference_model.nparams, verbose=self.verbose,
                                log_pdf_target=self.evaluate_log_posterior_data, **kwargs)
            self.samples = None

        elif self.sampling_method == 'IS':
            # importance distribution is either given by the user, or it is set as the prior of the model
            if 'proposal' not in kwargs or kwargs['proposal'] is None:
                if self.inference_model.prior is None:
                    raise NotImplementedError('A proposal density or a prior must be provided.')
                kwargs['proposal'] = self.inference_model.prior

            self.sampler = IS(log_pdf_target=self.evaluate_log_posterior_data, verbose=self.verbose, **kwargs)
            self.samples = None
            self.weights = None

        else:
            raise ValueError('Sampling_method should be either "MCMC" or "IS"')

        if self.verbose:
            print('Initialization of sampling technique completed successfully!')

        # Run the analysis if a certain number of samples was provided
        if (nsamples is not None) or (nsamples_per_chain is not None):
            self.run_estimation(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)

    def run_estimation(self, nsamples=None, nsamples_per_chain=None):
        if self.verbose:
            print('Running parameter estimation with ' + self.sampling_method + ' completed successfully!')

        if self.sampling_method == 'MCMC':
            self.sampler.run(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)
            self.samples = self.sampler.samples
        elif self.sampling_method == 'IS':
            if nsamples_per_chain is not None:
                raise ValueError('nsamples_per_chain is not an appropriate input for IS.')
            self.sampler.run(nsamples=nsamples)
            self.samples, self.weights = self.sampler.samples, self.sampler.weights

    def evaluate_log_posterior_data(self, params):
        return self.inference_model.evaluate_log_posterior(data=self.data, params=params)

########################################################################################################################
########################################################################################################################
#                                  Inference Model Selection
########################################################################################################################


class BayesModelSelection:

    def __init__(self, candidate_models, data, prior_probabilities=None, sorted_outputs=True,
                 method_evidence_computation='harmonic_mean', verbose=False, nsamples=None, nsamples_per_chain=None,
                 **kwargs):

        """
            Perform model selection using Inference criteria.

            Inputs:

                :param candidate_models: candidate models, must be a list of instances of class InferenceModel
                :type candidate_models: list

                :param data: available data
                :type data: ndarray

                :param prior_probabilities: prior probabilities of each model
                :type prior_probabilities: list of floats

                :param kwargs: inputs to the maximum likelihood estimator, for each model
            :type kwargs: dictionary, each value should be a list of length len(candidate_models)

                :param sorted_outputs: indicates if results are returned in sorted order, according to model
                 probabilities
                :type sorted_outputs: bool

            Outputs:

            A list of sorted models, their posterior probability based on data, the evidence of the model and the
            samples representing the posterior pdf.

            # Authors: Yuchen Zhou
            # Updated: 12/17/18 by Audrey Olivier

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
        self.sorted_outputs = sorted_outputs

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
            bayes_estimator.run_estimation(nsamples=nsamples, nsamples_per_chain=nsamples_per_chain)
            self.evidences[i] = self.estimate_evidence(
                inference_model=inference_model, posterior_samples=bayes_estimator.sampler.samples,
                log_posterior_values=bayes_estimator.sampler.log_pdf_values)

        # Compute posterior probabilities
        self.probabilities = self.compute_posterior_probabilities(evidence_values=self.evidences)

        # sort the models
        if self.sorted_outputs:
            sort_idx = list(np.argsort(np.array(self.probabilities)))[::-1]

            self.candidate_models = [self.candidate_models[i] for i in sort_idx]
            self.probabilities = [self.probabilities[i] for i in sort_idx]
            self.evidences = [self.evidences[i] for i in sort_idx]

        if self.verbose:
            print('Bayesian Model Selection analysis completed!')

    def estimate_evidence(self, inference_model, posterior_samples, log_posterior_values):
        """ Estimate evidence from MCMC samples, for one model """
        if self.method_evidence_computation.lower() == 'harmonic_mean':
            #samples[int(0.5 * len(samples)):]
            log_likelihood_values = log_posterior_values - inference_model.prior.log_pdf(x=posterior_samples)
            temp = np.mean(1./np.exp(log_likelihood_values))
        else:
            raise ValueError('Only the harmonic mean method is currently supported')
        return 1/temp

    def compute_posterior_probabilities(self, evidence_values):
        """ Compute the posterior probabilities, knowing the values of the evidence """
        scaled_evidences = [evi * prior_prob for (evi, prior_prob) in zip(evidence_values, self.prior_probabilities)]
        return scaled_evidences / np.sum(scaled_evidences)

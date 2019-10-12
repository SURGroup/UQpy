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
warnings.filterwarnings("ignore")


########################################################################################################################
########################################################################################################################
#                            Define the model - probability model or python model
########################################################################################################################

class InferenceModel:
    def __init__(self, n_params, run_model_object=None, log_likelihood=None, name='', error_covariance=1.0,
                 prior_name=None, prior_params=None, prior_copula=None, prior_copula_params=None,
                 verbose=False, likelihood_compute_multiple=True, kwargs_likelihood={}, kwargs_run_model={}
                 ):

        """
        Define the model, either as a python script, or a probability model

        Inputs:
            :param model_type: model type (python script or probability model)
            :type model_type: str, 'python' or 'pdf'

            :param model_name: name of the model, required only if model_type = 'pdf' (name of the distribution)
            :type model_name: str

            :param model_script: python script that encodes the model, required only if model_type = 'python'
            :type model_script: str, format '< >.py'

            :param n_params: number of parameters to be estimated, required
            :type n_params: int

            :param error_adapt: if set to True, the algorithm learns the covariance term,
                   can only be used when model is defined as a python script
            :type error_adapt: boolean

            :param error_covariance: covariance of the Gaussian error for model defined by a python script
            :type error_covariance: ndarray (full covariance matrix) or float (diagonal values)

            :param prior_name: distribution name of the prior pdf
            :type prior_name: str or list of str

            :param prior_params: parameters of the prior pdf
            :type prior_params: ndarray or list of ndarrays

            :param prior_copula: copula of the prior, if necessary
            :param prior_copula: str

            :param prior_copula_params: parameters of the copula of the prior, if necessary
            :param prior_copula_params: str

            :param log_likelihood: name of log-likelihood function. Default is None. The log-likelihood function takes
            the data, model prediction, and the corresponding model parameters as input. Any additional arguments can be
            passed through the dictionary kwargs.
            :type log_likelihood: function or None

            :param kwargs: all additional keyword arguments for the log-likelihood function, if necessary
            :type kwargs: dictionary

            :param model_object_name, input_template, var_names, output_script, output_object_name, ntasks,
                   cores_per_task, nodes, resume, verbose, model_dir, cluster: parameters of the python model, see
                   RunModel.py for explanations, required only if model_type == 'python'
            :param model_object_name, input_template, var_names, output_script, output_object_name, ntasks,
                   cores_per_task, nodes, resume, verbose, model_dir, cluster: see RunModel

        Output:

            :return Model.log_like: function that computes the log likelihood, given some data and a parameter value
            :rtype Model.log_like: method

            :return Model.prior: probability distribution of the prior
            :rtype Model.prior: instance from Distribution class

        """
        self.n_params = n_params
        if not isinstance(self.n_params, int) or self.n_params <= 0:
            raise TypeError('Input n_params must be an integer > 0.')
        self.name = name
        if not isinstance(self.name, str):
            raise TypeError('Input name must be a string.')
        self.verbose = verbose

        # The log-likelihood function will be defined via RunModel and / or the log_likelihood function - The following
        # lines define a function self.log_likelihood(data, model_output, params, **kwargs) for all cases.

        # Case 1: RunModel object exists
        if run_model_object is not None:
            if not isinstance(run_model_object, RunModel):   # must be of class RunModel
                raise TypeError('Input run_model_object should be an object of instance RunModel.')
            self.run_model_object = run_model_object
            self.kwargs_run_model = kwargs_run_model

            # Case 1.a: Gaussian error inference model
            if log_likelihood is None:
                if not isinstance(error_covariance, (float, int, np.ndarray)):
                    raise TypeError('Input error_covariance must be a float or 1D array or 2D array.')
                self.tmp_log_likelihood = lambda data, model_output, params, **kwargs: multivariate_normal.logpdf(
                    data, mean=np.array(model_output).reshape((-1, )), cov=error_covariance)
                self.kwargs_likelihood = {}
                self.likelihood_compute_multiple = False

            # Case 1.b: likelihood model is provided by the user
            else:
                self.tmp_log_likelihood = log_likelihood
                self.kwargs_likelihood = kwargs_likelihood
                self.likelihood_compute_multiple = likelihood_compute_multiple

        # Case 2: no RunModel object, the log_likelihood function will be fully defined by input log_likelihood
        else:
            self.run_model_object = None
            self.kwargs_run_model = None
            if log_likelihood is None:
                raise ValueError('A run_model_object or a log_likelihood input must be provided by the user.')

            # Case 2.a: self.log_likelihood is a string, defines a Distribution and use its log_pdf or pdf method
            if isinstance(log_likelihood, str):
                # Use either the log_pdf (preferred) or pdf method of Distribution()
                self.tmp_log_likelihood = None
                self.dist_log_likelihood = Distribution(log_likelihood)
                self.kwargs_likelihood = kwargs_likelihood
                self.likelihood_compute_multiple = False

            # Case 2.b: self.log_likelihood is a callable, just use it as is
            elif callable(log_likelihood):
                self.tmp_log_likelihood = log_likelihood
                self.kwargs_likelihood = kwargs_likelihood
                self.likelihood_compute_multiple = likelihood_compute_multiple

            else:
                raise TypeError('Input log_likelihood (without RunModel object) must be a callable or a string.')

        # Define prior if it is given
        if prior_name is not None:
            self.prior = Distribution(dist_name=prior_name, copula=prior_copula)
            self.kwargs_prior = {'params': prior_params, 'copula_params': prior_copula_params}
        else:
            self.prior = None

    def evaluate_log_likelihood(self, data, params):
        """ Computes the log-likelihood of model
            inputs: data, ndarray of dimension (ndata, )
                    params, ndarray of dimension (nsamples, n_params) or (n_params,)
            output: ndarray of size (nsamples, ), contains log likelihood of p(data | params[i,:])
        """
        if isinstance(params, list):
            params = np.array(params)
        if not isinstance(params, np.ndarray):
            raise TypeError('Input params should be an ndarray (preferred) or a list.')
        flag_one_input = False
        if len(params.shape) == 1:
            flag_one_input = True
            params = params.reshape((1, self.n_params))
        if params.shape[1] != self.n_params:
            raise ValueError('Wrong dimensions in params.')

        # Case 1.
        if self.run_model_object is not None:
            self.run_model_object.run(samples=params, **self.kwargs_run_model)
            # TODO: review this when modification to RunModel is performed
            model_outputs = self.run_model_object.qoi_list[-params.shape[0]:]
            if self.likelihood_compute_multiple:
                results = self.tmp_log_likelihood(data=data, model_output=model_outputs, params=params,
                                                  **self.kwargs_likelihood)
            else:
                results = np.array([self.tmp_log_likelihood(data=data, model_output=model_output_, params=params_,
                                                            **self.kwargs_likelihood)
                                    for (model_output_, params_) in zip(model_outputs, params)])
        # Case 2.a
        elif hasattr(self, 'dist_log_likelihood'):
            try:
                results = [np.sum(self.dist_log_likelihood.log_pdf(x=data, params=params_, **self.kwargs_likelihood))
                           for params_ in params]
            except AttributeError:
                try:
                    results = [
                        np.sum(np.log(self.dist_log_likelihood.pdf(x=data, params=params_, **self.kwargs_likelihood)))
                        for params_ in params]
                except AttributeError:
                    raise AttributeError('Input log_likelihood of InferenceModel provided as a string must point to a '
                                         'Distribution with an existing log_pdf or pdf method.')
            results = np.array(results)

        # Case 2.b
        else:
            if self.likelihood_compute_multiple:
                results = self.tmp_log_likelihood(data=data, params=params, **self.kwargs_likelihood)
            else:
                results = np.array([self.tmp_log_likelihood(data=data, params=params_, **self.kwargs_likelihood)
                                    for params_ in params])
        if flag_one_input:
            return results[0]
        return results

    def evaluate_log_posterior(self, data, params):
        """ Computes the log posterior (scaled): log[p(data|params) * p(params)]
            inputs: data, ndarray
                    params, ndarray of dimension (nsamples, n_params) or (n_params,)
            output: ndarray of size (nsamples, ), contains log likelihood of p(data | params[i,:])
        Note: if the Inference model does not possess a prior, an uninformatic prior p(params)=1 is assumed
        """
        if isinstance(params, list):
            params = np.array(params)
        if not isinstance(params, np.ndarray):
            raise TypeError('Input params should be an ndarray (preferred) or a list.')
        flag_one_input = False
        if len(params.shape) == 1:
            flag_one_input = True
            params = params.reshape((1, self.n_params))
        if params.shape[1] != self.n_params:
            raise ValueError('Wrong dimensions in params.')

        # If the prior is not provided it is set to an non-informative prior p(theta)=1, log_posterior = log_likelihood
        if self.prior is None:
            return self.evaluate_log_likelihood(data=data, params=params)

        # Otherwise, use prior provided in the InferenceModel setup
        try:
            log_prior_eval = self.prior.log_pdf(x=params, **self.kwargs_prior)
        except AttributeError:
            try:
                log_prior_eval = np.log(self.prior.pdf(x=params, **self.kwargs_prior))
            except AttributeError:
                raise AttributeError('The prior of InferenceModel must have a log_pdf or pdf method.')
        log_likelihood_eval = self.evaluate_log_likelihood(data=data, params=params)
        if flag_one_input:
            return (log_likelihood_eval + log_prior_eval)[0]
        return log_likelihood_eval + log_prior_eval


########################################################################################################################
########################################################################################################################
#                                  Maximum Likelihood Estimation
########################################################################################################################

class MLEstimation:

    def __init__(self, inference_model, data, verbose=False, iter_optim=1, **kwargs):

        """
        Perform maximum likelihood estimation, i.e., given some data y, compute the parameter vector that maximizes the
        likelihood p(y|theta).

        Inputs:
            :param model: the model
            :type model: instance of class Model

            :param data: Available data
            :type data: ndarray of size (ndata, )

            :param iter_optim: number of iterations for the maximization procedure
                (each iteration starts at a random point)
            :type iter_optim: an integer >= 1, default 1

            :param method_optim: method for optimization, see scipy.optimize.minimize
            :type method_optim: str

            :param bounds: bounds in each dimension
            :type bounds: list (of length n_params) of lists (each of dimension 2)

        Output:
            :return: MLEstimation.param: value of parameter vector that maximizes the likelihood
            :rtype: MLEstimation.param: ndarray

            :return: MLEstimation.max_log_like: value of the maximum likelihood
            :rtype: MLEstimation.max_log_like: float

        """
        self.inference_model = inference_model
        if not isinstance(inference_model, InferenceModel):
            raise TypeError('Input inference_model should be of type InferenceModel')
        self.data = data
        self.iter_optim = iter_optim
        if not isinstance(self.iter_optim, int) or self.iter_optim < 1:
            raise TypeError('Input iter_optim should be an integer greater than 1.')
        self.kwargs_optim = kwargs
        self.verbose = verbose

        # Case 2a
        if hasattr(self.inference_model, 'dist_log_likelihood'):
            try:
                param = np.array(self.inference_model.dist_log_likelihood.fit(self.data))
                max_log_like = self.inference_model.evaluate_log_likelihood(data=self.data, params=param)
                if verbose:
                    print('Evaluating maximum likelihood estimate for inference model ' + inference_model.name +
                          ', using fit method.')
            except AttributeError:
                if verbose:
                    print('Evaluating maximum likelihood estimate for inference model ' + inference_model.name)
                param, max_log_like = self.run_optimization()
        else:
            if verbose:
                print('Evaluating maximum likelihood estimate for inference model ' + inference_model.name)
            param, max_log_like = self.run_optimization()

        self.param = param
        self.max_log_like = max_log_like
        if verbose:
            print('Max likelihood estimation completed.')

    def run_optimization(self):

        list_param = []
        list_max_log_like = []
        if self.iter_optim > 1 or 'x0' not in self.kwargs_optim.keys():
            x0 = np.random.rand(self.iter_optim, self.inference_model.n_params)
            if 'bounds' in self.kwargs_optim.keys():
                bounds = np.array(self.kwargs_optim['bounds'])
                x0 = bounds[:, 0].reshape((1, -1)) + (bounds[:, 1]-bounds[:, 0]).reshape((1,-1)) * x0
            self.kwargs_optim['x0'] = x0
        else:
            self.kwargs_optim['x0'] = self.kwargs_optim['x0'].reshape((1,-1))

        for i in range(self.iter_optim):
            res = minimize(self.evaluate_neg_log_likelihood_data, **self.kwargs_optim)
            list_param.append(res.x)
            list_max_log_like.append((-1)*res.fun)
        idx_max = int(np.argmax(list_max_log_like))
        param = np.array(list_param[idx_max])
        max_log_like = list_max_log_like[idx_max]
        return param, max_log_like

    def evaluate_neg_log_likelihood_data(self, one_param):
        return -1 * self.inference_model.evaluate_log_likelihood(data=self.data, params=one_param)


########################################################################################################################
########################################################################################################################
#                                  Model Selection Using Information Theoretic Criteria
########################################################################################################################

class InfoModelSelection:

    def __init__(self, candidate_models, data, criterion=None, verbose=False, sorted_outputs=True,
                 **kwargs):

        """
            Perform model selection using information theoretic criteria. Supported criteria are BIC, AIC (default), AICc.

            Inputs:

            :param candidate_models: Candidate models, must be a list of instances of class Model
            :type candidate_models: list

            :param data: Available data
            :type data: ndarray

            :param criterion: Method to be used
            :type criterion: str

            :param method_optim, x0, iter_optim, bounds: inputs to the maximum likelihood estimator, for each model
            :type method_optim, x0, iter_optim, bounds: lists of length len(candidate_models), see MLEstimation

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
        self.data = data
        self.criterion = criterion

        if not all(isinstance(value, (list, tuple)) for (key, value) in kwargs.items()) or not all(
            len(value) == len(candidate_models) for (key, value) in kwargs.items()):
            print([(key, value) for (key, value) in kwargs.items()])
            raise TypeError('Extra inputs to model selection must be lists of length len(candidate_models)')

        # First evaluate ML estimate for all models
        fitted_params = []
        criteria = []
        penalty_terms = []
        for i, inference_model in enumerate(candidate_models):
            kwargs_i = dict([(key, value[i]) for (key, value) in kwargs.items()])
            ml_estimator = MLEstimation(inference_model=inference_model, data=self.data, verbose=verbose,
                                        **kwargs_i,
                                        )
            fitted_params.append(ml_estimator.param)
            max_log_like = ml_estimator.max_log_like

            k = inference_model.n_params
            n = np.size(data)
            if self.criterion == 'BIC':
                criterion_value = -2 * max_log_like + np.log(n) * k
                penalty_term = np.log(n) * k
            elif self.criterion == 'AICc':
                criterion_value = -2 * max_log_like + 2 * k + (2 * k ** 2 + 2 * k) / (n - k - 1)
                penalty_term = 2 * k + (2 * k ** 2 + 2 * k) / (n - k - 1)
            else: # default: do AIC
                criterion_value = -2 * max_log_like + 2 * k
                penalty_term = 2 * k
            criteria.append(criterion_value)
            penalty_terms.append(penalty_term)

        delta = np.array(criteria) - min(criteria)
        prob = np.exp(-delta / 2)
        probabilities = prob / np.sum(prob)

        # return outputs in sorted order, from most probable model to least probable model
        if sorted_outputs:
            sort_idx = list(np.argsort(np.array(criteria)))
        # or in initial order
        else:
            sort_idx = list(range(len(candidate_models)))
        self.candidate_models = [candidate_models[i] for i in sort_idx]
        self.model_names = [model.name for model in self.candidate_models]
        self.fitted_params = [fitted_params[i] for i in sort_idx]
        self.criterion_values = [criteria[i] for i in sort_idx]
        self.penalty_terms = [penalty_terms[i] for i in sort_idx]
        self.probabilities = [probabilities[i] for i in sort_idx]


########################################################################################################################
########################################################################################################################
#                                  Inference Parameter estimation
########################################################################################################################

class BayesParameterEstimation:

    def __init__(self, inference_model, data, sampling_method='MCMC', verbose=False, **kwargs):

        """
            Generates samples from the posterior distribution, using MCMC or IS.

            Inputs:

            :param inference_model: model, must be an instance of class Model
            :type inference_model: list

            :param data: Available data
            :type data: ndarray

            :param sampling_method: Method to be used
            :type sampling_method: str, 'MCMC' or 'IS'

            :param pdf_proposal_type, pdf_proposal_scale, pdf_proposal, pdf_proposal_params, algorithm, jump, nsamples, nburn,
             seed: inputs to the sampling method, see MCMC and IS
            :type pdf_proposal_type, pdf_proposal_scale, pdf_proposal, pdf_proposal_params, algorithm, jump, nsamples, nburn,
             seed: see MCMC and IS

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

        if self.sampling_method.lower() == 'mcmc':

            if verbose:
                print('Running Bayesian parameter estimation using MCMC for inference model '+self.inference_model.name)

            # If the seed is not provided, sample one from the prior pdf of the parameters
            if 'seed' not in kwargs.keys() or kwargs['seed'] is None:
                if self.inference_model.prior is None:
                    raise NotImplementedError('A prior with a rvs method or a seed must be provided for MCMC.')
                try:
                    nchains = 1
                    if 'nchains' in kwargs.keys() and kwargs['nchains'] is not None:
                        nchains = kwargs['nchains']
                    kwargs['seed'] = self.inference_model.prior.rvs(nsamples=nchains)
                except AttributeError:
                    raise NotImplementedError('A prior with a rvs method or a seed must be provided for MCMC.')

            dimension = self.inference_model.n_params
            z = MCMC(dimension=dimension, log_pdf_target=self.evaluate_log_posterior_data, **kwargs)

            self.samples = z.samples
            self.accept_ratio = z.accept_ratio

        elif self.sampling_method == 'IS':

            if verbose:
                print('Running Bayesian parameter estimation using IS for candidate model:', self.inference_model.name)

            # importance distribution is either given by the user, or it is set as the prior of the model
            if 'pdf_proposal' not in kwargs or kwargs['pdf_proposal'] is None:
                if self.inference_model.prior is None:
                    raise NotImplementedError('A proposal density or a prior (with an rvs method) must be given.')
                kwargs['pdf_proposal'] = self.inference_model.prior.dist_name
                kwargs['pdf_proposal_params'] = self.inference_model.kwargs_prior['params']

            z = IS(log_pdf_target=self.evaluate_log_posterior_data, **kwargs)

            self.samples = z.samples
            self.weights = z.weights

        else:
            raise ValueError('Sampling_method should be either "MCMC" or "IS"')

        if verbose:
            print('Parameter estimation analysis completed!')

    def evaluate_log_posterior_data(self, params):
        return self.inference_model.evaluate_log_posterior(data=self.data, params=params)

########################################################################################################################
########################################################################################################################
#                                  Inference Model Selection
########################################################################################################################


class BayesModelSelection:

    def __init__(self, candidate_models, data, prior_probabilities=None, sorted_outputs=True,
                 method_evidence_computation='harmonic_mean', verbose=False, **kwargs):

        """
            Perform model selection using Inference criteria.

            Inputs:

                :param candidate_models: candidate models, must be a list of instances of class Model
                :type candidate_models: list

                :param data: available data
                :type data: ndarray

                :param prior_probabilities: prior probabilities of each model
                :type prior_probabilities: list of floats

                :param pdf_proposal_type, pdf_proposal_scale, algorithm, jump, nsamples, nburn, seed:
                inputs to the sampling method, see MCMC
                :type pdf_proposal_type, pdf_proposal_scale, algorithm, jump, nsamples, nburn, seed:
                see MCMC - must be lists of values, of len = len(candidate_models)

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
        self.data = data
        if not all(isinstance(value, (list, tuple)) for (key, value) in kwargs.items()) or not all(
            len(value) == len(candidate_models) for (key, value) in kwargs.items()):
            print([(key, value) for (key, value) in kwargs.items()])
            raise TypeError('Extra inputs to model selection must be lists of length len(candidate_models)')
        self.kwargs = kwargs
        self.method_evidence_computation = method_evidence_computation

        self.verbose=verbose
        if prior_probabilities is None:
            self.prior_probabilities = [1/len(candidate_models) for _ in candidate_models]
        else:
            self.prior_probabilities = prior_probabilities

        model_probabilities, evidence, parameter_estimation = self.run_multi_bayes_ms()

        # sort the models
        if sorted_outputs:
            sort_idx = list(np.argsort(np.array(model_probabilities)))[::-1]
        else:
            sort_idx = list(range(len(candidate_models)))
        self.candidate_models = [candidate_models[i] for i in sort_idx]
        self.model_names = [model.name for model in self.candidate_models]
        self.mcmc_outputs = [parameter_estimation[i] for i in sort_idx]
        self.probabilities = [model_probabilities[i] for i in sort_idx]
        self.evidences = [evidence[i] for i in sort_idx]

    def run_multi_bayes_ms(self):

        if self.verbose:
            print('Running Bayesian Model Selection.')
        # Initialize the evidence or marginal likelihood
        evi_value = np.zeros((len(self.candidate_models),))
        scaled_evi = np.zeros_like(evi_value)
        pe = []
        # Perform MCMC for all candidate models
        for i, inference_model in enumerate(self.candidate_models):
            kwargs_i = dict([(key, value[i]) for (key, value) in self.kwargs.items()])
            if self.verbose:
                print('UQpy: Running MCMC for model '+inference_model.name)

            pe_i = BayesParameterEstimation(data=self.data, inference_model=inference_model,
                                            verbose=self.verbose, sampling_method='MCMC', **kwargs_i)
            pe.append(pe_i)
            if self.method_evidence_computation.lower() == 'harmonic_mean':
                evi_value[i] = self.estimate_evidence_HM(pe_i.samples, inference_model)
            else:
                raise ValueError('Currently, the only method supported for evidence computation in the Harmonic mean.')
            scaled_evi[i] = evi_value[i] * self.prior_probabilities[i]

        model_prob = scaled_evi / np.sum(scaled_evi)
        if self.verbose:
            print('Bayesian Model Selection analysis completed!')

        return model_prob, evi_value, pe

    def estimate_evidence_HM(self, samples, inference_model):
        # The method used here is the harmonic mean
        likelihood_given_sample = [inference_model.evaluate_log_likelihood(self.data, x)
                                   for x in samples[int(0.5 * len(samples)):]]
        temp = np.mean([1/np.exp(x) for x in likelihood_given_sample])
        return 1/temp

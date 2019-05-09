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

class Model:
    def __init__(self, model_type=None, model_script=None, model_name=None,
                 n_params=None, error_covariance=1.0,
                 prior_name=None, prior_params=None, prior_copula=None,
                 model_object_name=None, input_template=None, var_names=None, output_script=None,
                 output_object_name=None, ntasks=1, cores_per_task=1, nodes=1, resume=False,
                 model_dir=None, cluster=False, verbose=False,
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

            :param error_covariance: covariance of the Gaussian error for model defined by a python script
            :type error_covariance: ndarray (full covariance matrix) or float (diagonal values)

            :param cpu_runModel: number of cpus used when using runModel, used only when model_type='python'
            :type cpu_runModel: int

            :param prior_name: distribution name of the prior pdf
            :type prior_name: str or list of str

            :param prior_params: parameters of the prior pdf
            :type prior_params: ndarray or list of ndarrays

            :param prior_copula: copula of the prior, if necessary
            :param prior_copula: str

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
        self.type = model_type
        self.name = model_name
        if n_params is None:
            raise ValueError('A number of parameters must be defined.')
        self.n_params = n_params
        self.verbose = verbose

        if self.type == 'python':
            # Check that the script is a python file
            if not model_script.lower().endswith('.py'):
                raise ValueError('A python script, with extension .py, must be provided.')
            self.script = model_script
            self.model_object_name = model_object_name
            if self.name is None:
                self.name = self.script+self.model_object_name
            self.var_names = var_names
            if self.var_names is None:
                self.var_names = ['theta_{}'.format(i) for i in range(self.n_params)]
            self.error_covariance = error_covariance
            self.output_object_name = output_object_name
            self.input_template = input_template
            self.output_script = output_script
            self.ntasks = ntasks
            self.cores_per_task = cores_per_task
            self.nodes = nodes
            self.resume = resume
            self.model_dir = model_dir
            self.cluster = cluster

        elif self.type == 'pdf':
            self.pdf = Distribution(dist_name=self.name)

        else:
            raise ValueError('UQpy error: model_type must be defined, as either "pdf" of "python".')

        # Define prior if it is given
        if prior_name is not None:
            self.prior = Distribution(dist_name= prior_name, copula = prior_copula)
            self.prior_params = prior_params
        else:
            self.prior = None
            self.prior_params = None

    def log_like(self, data, params):
        """ Computes the log-likelihood of model
            inputs: data, ndarray of dimension (ndata, )
                    params, ndarray of dimension (nsamples, n_params) or (n_params,)
            output: ndarray of size (nsamples, ), contains log likelihood of p(data | params[i,:])
        """
        if params.size == self.n_params:
            params = params.reshape((1, self.n_params))
        if params.shape[1] != self.n_params:
            raise ValueError('the nb of columns in params should be equal to model.n_params')
        results = np.empty((params.shape[0],), dtype=float)
        if self.type == 'python':
            z = RunModel(samples=params, model_script=self.script, model_object_name=self.model_object_name,
                         input_template=self.input_template, var_names=self.var_names,
                         output_script=self.output_script, output_object_name=self.output_object_name,
                         ntasks=self.ntasks, cores_per_task=self.cores_per_task, nodes=self.nodes,
                         resume=self.resume, verbose=self.verbose, model_dir=self.model_dir,
                         cluster=self.cluster)
            for i in range(params.shape[0]):
                mean = np.array(z.qoi_list[i]).reshape((-1,))
                results[i] = multivariate_normal.logpdf(data, mean=mean, cov=self.error_covariance)
        elif self.type == 'pdf':
            for i in range(params.shape[0]):
                param_i = params[i, :].reshape((self.n_params,))
                results[i] = np.sum(self.pdf.log_pdf(data, param_i))
        return results


########################################################################################################################
########################################################################################################################
#                                  Maximum Likelihood Estimation
########################################################################################################################

class MLEstimation:

    def __init__(self, model=None, data=None, method_optim=None, x0=None, iter_optim=1,
                 bounds=None, verbose=False):

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

        if not isinstance(model, Model):
            raise ValueError('UQpy error: model should be of type Model')
        self.model = model
        self.data = data
        self.method_optim = method_optim
        if iter_optim is None:
            iter_optim = 1
        self.iter_optim = iter_optim
        self.x0 = x0
        self.bounds = bounds
        self.verbose = verbose

        if model.type == 'python':
            if verbose:
                print('Evaluating max likelihood estimate for model ' + model.name + ' using optimization.')
            param, max_log_like = self.max_by_optimization()
            self.param = param
            self.max_log_like = max_log_like

        elif model.type == 'pdf':
            # Use the fit method if it exists
            try:
                if verbose:
                    print('Evaluating max likelihood estimate for model ' + model.name + ' using its fit method.')
                self.param = np.array(model.pdf.fit(self.data))
                self.max_log_like = model.log_like(self.data, self.param)[0]
            # Else use the optimization procedure
            except AttributeError:
                if verbose:
                    print('Evaluating max likelihood estimate for model ' + model.name + ' using optimization.')
                param, max_log_like = self.max_by_optimization()
                self.param = param
                self.max_log_like = max_log_like

        if verbose:
            print('Max likelihood estimation completed.')

    def max_by_optimization(self):

        def neg_log_like_data(param):
            return -1 * self.model.log_like(self.data, param)[0]

        if self.verbose:
            print('Evaluating max likelihood estimate for model ' + self.model.name + ' using optimization procedure.')
        list_param = []
        list_max_log_like = []
        if self.iter_optim > 1 or self.x0 is None:
            x0 = np.random.rand(self.iter_optim, self.model.n_params)
            if self.bounds is not None:
                bounds = np.array(self.bounds)
                x0 = bounds[:,0].reshape((1,-1)) + (bounds[:,1]-bounds[:,0]).reshape((1,-1)) * x0
        else:
            x0 = self.x0.reshape((1,-1))
        # second case: use any other method that does not require a Jacobian
        # TODO: a maximization that uses a Jacobian which can be given analytically by user
        for i in range(self.iter_optim):
            res = minimize(neg_log_like_data, x0[i,:], method=self.method_optim, bounds=self.bounds)
            list_param.append(res.x)
            list_max_log_like.append((-1)*res.fun)
        idx_max = int(np.argmax(list_max_log_like))
        param = np.array(list_param[idx_max])
        max_log_like = list_max_log_like[idx_max]
        return param, max_log_like


########################################################################################################################
########################################################################################################################
#                                  Model Selection Using Information Theoretic Criteria
########################################################################################################################

class InfoModelSelection:

    def __init__(self, candidate_models=None, data=None, method=None, method_optim=None, x0=None, iter_optim=None,
                 bounds=None, verbose=False, sorted_outputs=True):

        """
            Perform model selection using information theoretic criteria. Supported criteria are BIC, AIC (default), AICc.

            Inputs:

            :param candidate_models: Candidate models, must be a list of instances of class Model
            :type candidate_models: list

            :param data: Available data
            :type data: ndarray

            :param method: Method to be used
            :type method: str

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
        if (candidate_models is None) or (not isinstance(candidate_models, list)):
            raise ValueError('A list of models is required.')
        if any(not isinstance(model, Model) for model in candidate_models):
            raise ValueError('All candidate models should be of type Model.')
        if data is None:
            raise ValueError('data must be provided')
        self.data = data
        self.method = method
        input_args = [method_optim, x0, iter_optim, bounds]
        for input_arg in input_args:
            if input_arg is None:
                continue
            if (not isinstance(input_arg, list)) or (len(input_arg)!=len(candidate_models)):
                raise ValueError('UQpy error: input argument should be given as a list of len=nb of models.')

        # First evaluate ML estimate for all models
        fitted_params = []
        criteria = []
        penalty_terms = []
        for i, model in enumerate(candidate_models):
            ml_estimator = MLEstimation(model=model, data=self.data, verbose=verbose,
                                        method_optim=(None if method_optim is None else method_optim[i]),
                                        x0=(None if x0 is None else x0[i]),
                                        iter_optim=(None if iter_optim is None else iter_optim[i]),
                                        bounds=(None if bounds is None else bounds[i]),
                                        )
            fitted_params.append(ml_estimator.param)
            max_log_like = ml_estimator.max_log_like

            k = model.n_params
            n = np.size(data)
            if self.method == 'BIC':
                criterion_value = -2 * max_log_like + np.log(n) * k
                penalty_term = np.log(n) * k
            elif self.method == 'AICc':
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
            self.sorted_models = [candidate_models[i] for i in sort_idx]
            self.sorted_model_names = [model.name for model in self.sorted_models]
            self.sorted_fitted_params = [fitted_params[i] for i in sort_idx]
            self.sorted_criteria = [criteria[i] for i in sort_idx]
            self.sorted_penalty_terms = [penalty_terms[i] for i in sort_idx]
            self.sorted_probabilities = [probabilities[i] for i in sort_idx]
        # or return the outputs in the order they were given
        else:
            self.models = candidate_models
            self.model_names = [model.name for model in self.models]
            self.fitted_params = fitted_params
            self.criteria = criteria
            self.penalty_terms = penalty_terms
            self.probabilities = list(probabilities)


########################################################################################################################
########################################################################################################################
#                                  Bayesian Parameter estimation
########################################################################################################################

class BayesParameterEstimation:

    def __init__(self, model=None, data=None, sampling_method=None,
                 pdf_proposal=None, pdf_proposal_scale=None, pdf_proposal_params=None,
                 algorithm=None, jump=None, nsamples=None, nburn=None, seed=None, verbose=False):

        """
            Generates samples from the posterior distribution, using MCMC or IS.

            Inputs:

            :param model: model, must be an instance of class Model
            :type model: list

            :param data: Available data
            :type data: ndarray

            :param sampling_method: Method to be used
            :type sampling_method: str, 'MCMC' or 'IS'

            :param pdf_proposal, pdf_proposal_scale, pdf_proposal_params, algorithm, jump, nsamples, nburn,
             seed: inputs to the sampling method, see MCMC and IS
            :type pdf_proposal, pdf_proposal_scale, pdf_proposal_params, algorithm, jump, nsamples, nburn,
             seed: see MCMC and IS

            Outputs:

            Attributes of bayes = BayesParameterEstimation(...). For MCMC, bayes.samples are samples from the posterior
             pdf. For IS, bayes.samples in combination with bayes.weights provide an estimate of the posterior.

        """

        if not isinstance(model, Model):
            raise ValueError('model should be of type Model')
        if data is None:
            raise ValueError('data should be provided')
        if nsamples is None:
            raise ValueError('nsamples should be defined')

        self.data = data
        self.nsamples = nsamples
        self.sampling_method = sampling_method
        self.model = model

        if self.sampling_method == 'MCMC':

            if verbose:
                print('UQpy: Running parameter estimation for candidate model ', model.name)

            self.seed = seed
            if self.seed is None:
                mle = MLEstimation(model=self.model, data=self.data, verbose=verbose)
                self.seed = mle.param

            z = MCMC(dimension=self.model.n_params, pdf_proposal_type=pdf_proposal,
                     pdf_proposal_scale=pdf_proposal_scale,
                     algorithm=algorithm, jump=jump, seed=self.seed, nburn=nburn,
                     nsamples=self.nsamples, pdf_target=self.posterior, log_pdf_target=self.log_posterior)

            if verbose:
                print('UQpy: Parameter estimation analysis completed!')

            self.samples = z.samples
            self.accept_ratio = z.accept_ratio

        elif self.sampling_method == 'IS':

            # importance distribution is either given by the user, or it is set as the prior of the model
            if pdf_proposal is None:
                if self.model.prior is None:
                    raise ValueError('a proposal density or a prior should be given')
                pdf_proposal = self.model.prior.name
                pdf_proposal_params = self.model.prior_params

            if verbose:
                print('UQpy: Running parameter estimation for candidate model:', model.name)

            z = IS(nsamples=self.nsamples,
                   pdf_proposal=pdf_proposal, pdf_proposal_params=pdf_proposal_params,
                   log_pdf_target=self.log_posterior)

            print('UQpy: Parameter estimation analysis completed!')

            self.samples = z.samples
            self.weights = z.weights

        else:
            raise ValueError('Sampling_method should be either "MCMC" or "IS"')

        del self.model

    def posterior(self, theta, params=None, copula_params=None):
        if type(theta) is not np.ndarray:
            theta = np.array(theta)
        if len(theta.shape) == 1:
            theta = theta.reshape((1,np.size(theta)))
        # non-informative prior, p(theta)=1 everywhere
        if self.model.prior is None:
            return np.exp(self.model.log_like(data=self.data, params=theta))
        # prior is given
        else:
            return np.exp(self.model.log_like(data=self.data, params=theta) +
                          self.model.prior.log_pdf(x=theta, params=self.model.prior_params))

    def log_posterior(self, theta, params=None, copula_params=None):
        if type(theta) is not np.ndarray:
            theta = np.array(theta)
        if len(theta.shape) == 1:
            theta = theta.reshape((1,np.size(theta)))
        # non-informative prior, p(theta)=1 everywhere
        if self.model.prior is None:
            return self.model.log_like(data=self.data, params=theta)
        # prior is given
        else:
            return self.model.log_like(data=self.data, params=theta) + \
                   self.model.prior.log_pdf(x=theta, params=self.model.prior_params)

########################################################################################################################
########################################################################################################################
#                                  Bayesian Model Selection
########################################################################################################################


class BayesModelSelection:

    def __init__(self, candidate_models=None, data=None, prior_probabilities=None,
                 pdf_proposal_type=None, pdf_proposal_scale=None, algorithm=None, jump=None, nsamples=None,
                 nburn=None, method=None, seed=None, sorted_outputs = True, verbose=False):

        """
            Perform model selection using Bayesian criteria.

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
        if (candidate_models is None) or (not isinstance(candidate_models, list)):
            raise ValueError('A list of models is required.')
        if any(not isinstance(model, Model) for model in candidate_models):
            raise ValueError('All candidate models should be of type Model.')
        else:
            self.candidate_models = candidate_models
        if data is None:
            raise ValueError('data must be provided')
        self.data = data
        for input_item in [algorithm, jump, nsamples, nburn, seed, pdf_proposal_type, pdf_proposal_scale]:
            if input_item is None:
                continue
            if (not isinstance(input_item, list)) or (len(input_item)!=len(candidate_models)):
                raise ValueError('Inputs of model selection using MCMC nust be given as lists of len=nb of models.')
        self.pdf_proposal_type = pdf_proposal_type
        self.pdf_proposal_scale = pdf_proposal_scale
        self.algorithm = algorithm
        self.jump = jump
        self.nsamples = nsamples
        self.nburn = nburn
        self.seed = seed
        self.tmp_candidate_model = None
        self.verbose=verbose

        if prior_probabilities is None:
            self.prior_probabilities = [1/len(self.candidate_models) for _ in self.candidate_models]
        else:
            self.prior_probabilities = prior_probabilities

        model_probabilities, evidence, parameter_estimation = self.run_multi_bayes_ms()

        # sort the models
        if sorted_outputs:
            sort_idx = list(np.argsort(np.array(model_probabilities)))[::-1]
            self.sorted_models = [candidate_models[i] for i in sort_idx]
            self.sorted_model_names = [model.name for model in self.sorted_models]
            self.sorted_mcmc_outputs = [parameter_estimation[i] for i in sort_idx]
            self.sorted_probabilities = [model_probabilities[i] for i in sort_idx]
            self.sorted_evidences = [evidence[i] for i in sort_idx]
        else:
            self.models = candidate_models
            self.model_names = [model.name for model in self.models]
            self.mcmc_outputs = parameter_estimation
            self.probabilities = model_probabilities
            self.evidences = evidence

    def run_multi_bayes_ms(self):

        if self.verbose:
            print('UQpy: Running Bayesian MS...')
        # Initialize the evidence or marginal likelihood
        evi_value = np.zeros((len(self.candidate_models),))
        scaled_evi = np.zeros_like(evi_value)
        pe = [0]*len(self.candidate_models)
        # Perform MCMC for all candidate models
        for i in range(len(self.candidate_models)):
            self.tmp_candidate_model = self.candidate_models[i]
            if self.verbose:
                print('UQpy: Running MCMC for model '+self.tmp_candidate_model.name)

            pe_i = BayesParameterEstimation(data=self.data, model=self.tmp_candidate_model, verbose=self.verbose,
                                            sampling_method = 'MCMC',
                                            pdf_proposal = (None if self.pdf_proposal_type is None
                                                            else self.pdf_proposal_type[i]),
                                            pdf_proposal_scale=(None if self.pdf_proposal_scale is None
                                                                else self.pdf_proposal_scale[i]),
                                            algorithm=(None if self.algorithm is None else self.algorithm[i]),
                                            jump=(None if self.jump is None else self.jump[i]),
                                            nsamples=(None if self.nsamples is None else self.nsamples[i]),
                                            nburn=(None if self.nburn is None else self.nburn[i]),
                                            seed=(None if self.seed is None else self.seed[i])
                                            )
            pe[i] = pe_i
            evi_value[i] = self.estimate_evidence(pe_i.samples)
            scaled_evi[i] = evi_value[i]*self.prior_probabilities[i]

        sum_evi_value = np.sum(scaled_evi)
        model_prob = scaled_evi / sum_evi_value
        if self.verbose:
            print('UQpy: Bayesian MS analysis completed!')

        return model_prob, evi_value, pe

    def estimate_evidence(self,samples):
        # The method used here is the harmonic mean
        likelihood_given_sample = [self.tmp_candidate_model.log_like(self.data, x)
                                   for x in samples[int(0.5 * len(samples)):]]
        temp = np.mean([1/np.exp(x) for x in likelihood_given_sample])
        return 1/temp
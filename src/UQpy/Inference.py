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
from scipy import integrate
from scipy.special import gamma
from scipy.stats import multivariate_normal, chi2, norm
from UQpy.RunModel import RunModel
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")


########################################################################################################################
########################################################################################################################
#                            Information theoretic model selection - AIC, BIC
########################################################################################################################

class Model:
    def __init__(self, model_type = None, model_script = None, model_name = None,
                 n_params = None, error_covariance = 1, verbose=False,
                 prior_name = None, prior_params=None, prior_copula=None,
                 model_object_name=None, input_template=None, var_names=None, output_script=None,
                 output_object_name=None, ntasks=1, cores_per_task=1, nodes=1, resume=False,
                 model_dir=None, cluster=False,
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

            :param n_params: number of parameters, required only if model_type == 'python'
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
        self.script = model_script
        self.name = model_name
        self.n_params = n_params
        self.error_covariance = error_covariance
        self.verbose = verbose

        if self.type == 'python':
            # Check that the script is a python file
            if not self.script.lower().endswith('.py'):
                raise ValueError('A python script, with extension .py, must be provided.')
            if self.n_params is None:
                raise ValueError('A number of parameters must be defined.')
            else:
                self.n_params = n_params
            if self.name is None:
                self.name = self.script
            self.var_names = var_names
            if self.var_names is None:
                self.var_names = ['theta_{}'.format(i) for i in range(self.n_params)]
            self.model_object_name = model_object_name
            self.input_template = input_template
            self.output_script = output_script
            self.output_object_name = output_object_name
            self.ntasks = ntasks
            self.cores_per_task = cores_per_task
            self.nodes = nodes
            self.resume = resume
            self.model_dir = model_dir
            self.cluster = cluster

        elif self.type == 'pdf':
            self.pdf = Distribution(self.name)
            self.n_params = self.pdf.n_params

        else:
            raise ValueError('UQpy error: model_type must be defined, as either "pdf" of "python".')

        # Define prior if it is given
        self.prior_params = prior_params
        if prior_name is not None:
            self.prior = Distribution(name = prior_name, copula = prior_copula)
            self.prior_params = prior_params
        else:
            self.prior = None
            self.prior_params = None

    def log_like(self, x, params):
        if np.size(params) == self.n_params:
            params = params.reshape((1, self.n_params))
        if params.shape[1] != self.n_params:
            raise ValueError('the nb of columns in params should be equal to model.n_params')
        results = np.empty((params.shape[0],), dtype=float)
        if self.type == 'python':
            with suppress_stdout():  # disable printing output comments
                z = RunModel(samples=params, model_script=self.script, model_object_name=self.model_object_name,
                             input_template=self.input_template, var_names=self.var_names,
                             output_script=self.output_script, output_object_name=self.output_object_name,
                             ntasks=self.ntasks, cores_per_task=self.cores_per_task, nodes=self.nodes,
                             resume=self.resume, verbose=self.verbose, model_dir=self.model_dir,
                             cluster=self.cluster)
            for i in range(params.shape[0]):
                mean = z.qoi_list[i].reshape((-1,))
                results[i] = multivariate_normal.logpdf(x, mean=mean, cov=self.error_covariance)
        elif self.type == 'pdf':
            for i in range(params.shape[0]):
                param_i = params[i, :].reshape((self.n_params,))
                results[i] = np.sum(self.pdf.log_pdf(x, param_i))
        return results

class MLEstimation:

    def __init__(self, model = None, data=None, iter_optim = 1, method_optim = 'nelder-mead', verbose=False):

        """
        Perform maximum likelihood estimation, i.e., given some data y, compute the parameter vector that maximizes the
        likelihood p(y|theta).

        Inputs:
            :param model: the model
            :type model: instance of class Model

            :param data: Available data
            :type data: ndarray

            :param iter_optim: number of iterations for the maximization procedure (each iteration starts at a random point)
            :type iter_optim: an integer >= 1, default 1

            :param method_optim: method for optimization, see scipy.optimize.minimize
            :type method_optim: str

        Output:
            :return: MLEstimation.param: value of parameter vector that maximizes the likelihood
            :rtype: MLEstimation.param: ndarray

            :return: MLEstimation.max_log_like: value of the maximum likelihood
            :rtype: MLEstimation.max_log_like: float

        """

        if not isinstance(model, Model):
            raise ValueError('UQpy error: model should be of type Model')
        self.data = data

        if model.type == 'python':

            def log_like_data(param):
                return -1*model.log_like(self.data, param)[0]
            if verbose:
                print('Evaluating max likelihood estimate for model '+model.name)
            list_param = []
            list_max_log_like = []
            for _ in range(iter_optim):
                x0 = np.random.rand(1, model.n_params)
                print(x0.shape)
                res = minimize(log_like_data, x0, method=method_optim,options = {'disp': True})
                list_param.append(res.x)
                list_max_log_like.append((-1)*res.fun)
            idx_max = int(np.argmax(list_max_log_like))
            self.param = np.array(list_param[idx_max])
            self.max_log_like = list_max_log_like[idx_max]

        elif model.type == 'pdf':
            if verbose:
                print('Evaluating max likelihood estimate for model '+model.name)
            self.param = np.array(model.pdf.fit(self.data))
            self.max_log_like = model.log_like(self.data, self.param)[0]


class InfoModelSelection:

    def __init__(self, candidate_models=None, data=None, method=None, verbose=False):

        """
        Perform model selection using information theoretic criteria. Supported criteria are BIC, AIC (default), AICc.

        Inputs:

        :param candidate_models: Candidate models, must be a list of instances of class Model
        :type candidate_models: list

        :param data: Available data
        :type data: ndarray

        :param method: Method to be used
        :type method: str

        Outputs:

        A list of sorted models, their probability based on data and the given criterion, and the parameters that
        maximize the log likelihood. The penalty term (Ockam razor) is also given.

        """

        self.data = data
        self.method = method

        # Check that all candidate models are of class Model, and that they are all of the same type, pdf or python
        all_notisinstance = [not isinstance(model, Model) for model in candidate_models]
        if any(all_notisinstance):
            raise ValueError('All candidate models should be of type Model.')

        all_typesnotequal = [model.type != candidate_models[0].type for model in candidate_models]
        if any(all_typesnotequal):
            raise ValueError('All candidate models should be of same type, pdf or python.')

        # First evaluate ML estimate for all models
        list_params = []
        list_criteria = []
        list_penalty_term = []
        for i, model in enumerate(candidate_models):
            ml_estimator = MLEstimation(model=model, data=self.data, verbose=verbose)
            list_params.append(ml_estimator.param)
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
            list_criteria.append(criterion_value)
            list_penalty_term.append(penalty_term)

        sort_idx = list(np.argsort(np.array(list_criteria)))
        self.sorted_models = [candidate_models[i] for i in sort_idx]
        self.sorted_params = [list_params[i] for i in sort_idx]
        self.sorted_criteria = [list_criteria[i] for i in sort_idx]
        self.sorted_penalty_terms = [list_penalty_term[i] for i in sort_idx]

        sorted_criteria = np.array(self.sorted_criteria)
        delta = sorted_criteria-sorted_criteria[0]
        prob = np.exp(-delta/2)
        self.sorted_probabilities = prob/np.sum(prob)

        self.sorted_names = [model.name for model in self.sorted_models]


########################################################################################################################
########################################################################################################################
#                                  Bayesian Parameter estimation
########################################################################################################################
class BayesParameterEstimation:

    def __init__(self, data=None, model=None, sampling_method = None,
                 pdf_proposal = None, pdf_proposal_scale=None, pdf_proposal_params = None,
                 algorithm=None, jump=None, nsamples=None, nburn=None, seed=None, verbose=False):

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

            z = IS(dimension=self.model.n_params, nsamples=self.nsamples,
                   pdf_proposal=pdf_proposal, pdf_proposal_params=pdf_proposal_params,
                   log_pdf_target=self.log_posterior)

            print('UQpy: Parameter estimation analysis completed!')

            self.samples = z.samples
            self.weights = z.weights

        else:
            raise ValueError('Sampling_method should be either "MCMC" or "IS"')

        del self.model

    def posterior(self, theta, params):
        if type(theta) is not np.ndarray:
            theta = np.array(theta)
        if len(theta.shape) == 1:
            theta = theta.reshape((1,np.size(theta)))
        # non-informative prior, p(theta)=1 everywhere
        if self.model.prior is None:
            return np.exp(self.model.log_like(x=self.data, params=theta))
        # prior is given
        else:
            return np.exp(self.model.log_like(x=self.data, params=theta) +
                          self.model.prior.log_pdf(x=theta, params=self.model.prior_params))

    def log_posterior(self, theta, params):
        if type(theta) is not np.ndarray:
            theta = np.array(theta)
        if len(theta.shape) == 1:
            theta = theta.reshape((1,np.size(theta)))
        # non-informative prior, p(theta)=1 everywhere
        if self.model.prior is None:
            return self.model.log_like(x=self.data, params=theta)
        # prior is given
        else:
            return self.model.log_like(x=self.data, params=theta) + \
                   self.model.prior.log_pdf(x=theta, params=self.model.prior_params)

########################################################################################################################
########################################################################################################################
#                                  Bayesian Inference
########################################################################################################################

class BayesModelSelection:

    def __init__(self, data=None, dimension=None, candidate_model=None, pdf_proposal_type=None, pdf_proposal_scale=None,
                 algorithm=None, jump=None, nsamples=None, nburn=None,  prior_probabilities=None,
                 param_prior_dist=None, prior_hyperparams=None, method=None, walkers=None):

        """

        :param data:
        :param dimension:
        :param candidate_model:
        :param pdf_proposal_type:
        :param pdf_proposal_scale:
        :param algorithm:
        :param jump:
        :param nsamples:
        :param nburn:
        :param prior_probabilities:
        :param prior_dist:
        :param prior_dist_params:

        """

        self.data = data
        self.dimension = dimension
        self.pdf_proposal_type = pdf_proposal_type
        self.pdf_proposal_scale = pdf_proposal_scale
        self.algorithm = algorithm
        self.jump = jump
        self.nsamples = nsamples
        self.nburn = nburn
        self.walkers = walkers

        if candidate_model is None:
            raise RuntimeError('A probability model is required.')
        else:
            self.candidate_model = candidate_model

        if param_prior_dist is None and prior_hyperparams is None:
            self.param_prior_dist = [0] * len(param_prior_dist)
            self.prior_hyperparams = [0] * len(prior_hyperparams)
            for i in range(len(param_prior_dist)):
                self.prior_distribution[i] = 'uniform'
                self.prior_hyperparams[i] = [0, 10**6]

        if prior_hyperparams is None and param_prior_dist is not None:
            raise RuntimeError('The parameters of the prior distribution models should be provided.')

        if prior_hyperparams is not None:
            self.prior_hyperparams = [0] * len(prior_hyperparams)
            for i in range(len(prior_hyperparams)):
                self.prior_hyperparams[i] = prior_hyperparams[i]

        if param_prior_dist is not None:
            self.param_prior_dist = [0]*len(param_prior_dist)
            for i in range(len(param_prior_dist)):
                self.param_prior_dist[i] = param_prior_dist[i]

        if method is None:
            self.method = 'MLE'
        else:
            self.method = method

        if prior_probabilities is None:
            self.prior_probabilities = [0]*len(self.candidate_model)
            for i in range(len(self.candidate_model)):
                self.prior_probabilities[i] = 1/len(self.candidate_model)
        else:
            self.prior_probabilities = prior_probabilities

        self.model_probabilities, self.evidence, self.parameter_estimation = self.run_multi_bayes_ms()

    def run_multi_bayes_ms(self):
        # Initialize the evidence or marginal likelihood
        evi_value = np.zeros(len(self.candidate_model))
        scaled_evi = np.zeros_like(evi_value)
        pe = [0]*len(self.candidate_model)
        for i in range(len(self.candidate_model)):
            print('UQpy: Running Bayesian MS...')
            self.tmp_candidate_model = Distribution(self.candidate_model[i])
            self.tmp_params_prior_dist = Distribution(self.param_prior_dist[i], self.prior_hyperparams[i])

            pe[i] = BayesParameterEstimation(data=self.data, dist=self.tmp_candidate_model,
                                             param_prior_dist=self.tmp_params_prior_dist.name,
                                             prior_hyperparams=self.tmp_params_prior_dist.params, method=self.method,
                                             pdf_proposal_type=self.pdf_proposal_type,
                                             pdf_proposal_scale=self.pdf_proposal_scale,
                                             algorithm=self.algorithm, jump=self.jump, nsamples=self.nsamples,
                                             nburn=self.nburn,
                                             walkers=self.walkers)
            samples = pe[i].samples

            if self.tmp_candidate_model.n_params == 2:
                print('UQpy: Solving integral for the evidence estimation...')
                x_lim, y_lim = zip(samples.min(0), samples.max(0))
                z1, err_z1 = self.integrate_posterior_2d(x_lim, y_lim)
            elif self.tmp_candidate_model.n_params == 3:
                print('UQpy: Solving integral for the evidence estimation...')
                x_lim, y_lim, z_lim = zip(samples.min(0), samples.max(0))
                z1, err_z1 = self.integrate_posterior_3d(x_lim, y_lim, z_lim)

            evi_value[i] = z1
            scaled_evi[i] = evi_value[i]*self.prior_probabilities[i]

        print('UQpy: Bayesian MS analysis completed!')
        sum_evi_value = np.sum(evi_value)
        model_prob = scaled_evi / sum_evi_value

        return model_prob, evi_value, pe

    def log_likelihood_x_prior(self, theta, args):
        _ = args
        log_prior_ = self.tmp_params_prior_dist.log_pdf(self.data, theta)
        log_like = self.tmp_candidate_model.log_pdf(self.data, theta)

        return np.sum(log_prior_ + log_like)

    def integrate_posterior_2d(self, x_lim, y_lim):
        fun_ = lambda theta2, theta1: np.exp(self.log_likelihood_x_prior([theta1, theta2], None))
        return integrate.dblquad(fun_, x_lim[0], x_lim[1], lambda x: y_lim[0], lambda x: y_lim[1])

    def integrate_posterior_3d(self, x_lim, y_lim, z_lim):
        fun_ = lambda theta0, theta1, theta2: np.exp(self.log_likelihood_x_prior([theta0, theta1, theta2], None))
        return integrate.tplquad(fun_, x_lim[0], x_lim[1], lambda x: y_lim[0], lambda x: y_lim[1],
                                 lambda x, y: z_lim[0], lambda x, y: z_lim[1])


########################################################################################################################
########################################################################################################################
#                                  Supporting functions
########################################################################################################################
def non_info_prior_dist(name, params):

    if name.lower() == 'normal' or name.lower() == 'gaussian':

        if params[1] < 0:
            return np.inf
        else:
            return 0.0

    elif name.lower() == 'uniform':

        if params[1] < params[0]:
            return np.inf
        else:
            return 0.0

    elif name.lower() == 'beta':

        if params[0] > 0 and params[1] > 0:
            return 0.0
        else:
            return np.inf

    elif name.lower() == 'gumberl_r':

        if params[1] < 0:
            return np.inf
        else:
            return 0.0

    elif name.lower() == 'chisquare':

        if params[1] < 0:
            return np.inf
        else:
            return 0.0

    elif name.lower() == 'lognormal':

        if params[1] < 0:
            return np.inf
        else:
            return 0.0

    elif name.lower() == 'gamma':

        if params[0] < 0 or params[2] < 0:
            return np.inf
        else:
            return 0.0

    elif name.lower() == 'exponential':

        if params[1] < 0:
            return np.inf
        else:
            return 0.0

    elif name.lower() == 'cauchy':

        if params[1] < 0:
            return np.inf
        else:
            return 0.0

    elif name.lower() == 'inv_gauss':

        if params[0] < 0.0028 or params[2]< 0:
            return np.inf
        else:
            return 0.0

    elif name.lower() == 'logistic':

        if params[1] < 0:
            return np.inf
        else:
            return 0.0

    elif name.lower() == 'pareto':

        if params[0] < 0 or params[2] < 0:
            return np.inf
        else:
            return 0.0

    elif name.lower() == 'rayleigh':

        if params[1] < 0:
            return np.inf
        else:
            return 0.0

    elif name.lower() == 'levy':

        if params[1] < 0:
            return np.inf
        else:
            return 0.0

    elif name.lower() == 'laplace':

        if params[1] < 0:
            return np.inf
        else:
            return 0.0

    elif name.lower() == 'maxwell':

        if params[1] < 0:
            return np.inf
        else:
            return 0.0
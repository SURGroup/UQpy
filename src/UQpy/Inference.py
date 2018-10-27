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
from scipy.stats import multivariate_normal
from UQpy.RunModel import RunModel
from scipy.optimize import minimize
import warnings
from UQpy.SampleMethods import MCMC
warnings.filterwarnings("ignore")


########################################################################################################################
########################################################################################################################
#                            Information theoretic model selection - AIC, BIC
########################################################################################################################

class Model:
    def __init__(self, model_type = None, model_script = None, model_name = None,
                 n_params = None, error_covariance = 1, cpu_RunModel = 1):

        """

        :param name: name of the model
        :type data: str

        :param param: parameter vector
        :type data: ndarray

        :param log_like_func: function that evaluates the log_likelihood, takes as an input the data and parameter vector
        :type log_like_func: function

        :param ML_fit: function that computes the maximum likelihood parameter estimate, takes as an input the data
        :type method: function

        """
        self.type = model_type
        self.script = model_script
        self.name = model_name
        self.n_params = n_params
        self.error_covariance = error_covariance
        self.cpu_RunModel = cpu_RunModel

        if self.type == 'python':
            # Check that the script is a python file
            if not self.script.lower().endswith('.py'):
                raise ValueError('A python script, with extension .py, must be provided.')
            if n_params is None:
                raise ValueError('A number of parameters must be defined.')
            else:
                self.n_params = n_params
            self.log_like = partial(self.log_like_normal)

        elif self.type == 'pdf':
            supported_distributions = get_supported_distributions(print_=False)
            if not self.name in supported_distributions:
                raise ValueError('probability distribution is not supported')
            ### what if custom distribution????
            self.pdf = Distribution(self.name)
            self.n_params = self.pdf.n_params
            self.log_like = self.log_like_sum

    def log_like_normal(self, x, params):
        params = params.reshape((1, self.n_params))
        with suppress_stdout():  # disable output
            z = RunModel(cpu=self.cpu_RunModel, model_type=self.type, model_script=self.script, dimension=self.n_params,
                         samples=params)
        mean = z.model_eval.QOI[0]
        return multivariate_normal.logpdf(x, mean=mean, cov=self.error_covariance)

    def log_like_sum(self, x, params):
        return np.sum(self.pdf.log_pdf(x, params))


class MLEstimation:

    def __init__(self, model_instance = None, data=None, iter_optim = 1, method_optim = 'nelder-mead'):

        if not isinstance(model_instance, Model):
            raise ValueError('model_instance should be of type Model')
        self.data = data

        if model_instance.type == 'python':

            def log_like_data(param):
                return -1*model_instance.log_like(self.data, param)
            print('Evaluating max likelihood estimate...')
            list_param = []
            list_max_log_like = []
            for _ in range(iter_optim):
                x0 = np.random.rand(1, model_instance.n_params)
                res = minimize(log_like_data, x0, method=method_optim,options = {'disp': True})
                list_param.append(res.x)
                list_max_log_like.append((-1)*res.fun)
            idx_max = int(np.argmax(list_max_log_like))
            self.param = np.array(list_param[idx_max])
            self.max_log_like = list_max_log_like[idx_max]

        elif model_instance.type == 'pdf':
            print('Evaluating max likelihood estimate...')
            self.param = model_instance.pdf.fit(self.data)
            self.max_log_like = model_instance.log_like(self.data, self.param)


class InfoModelSelection:

    def __init__(self, candidate_models=None, data=None, method=None):

        """

        :param data: Available data
        :type data: ndarray

        :param candidate_models: Candidate models, must be a list of instances of class Model
        :type candidate_models: list

        :param method: Method to be used
        :type method: str

        """

        self.data = data
        self.method = method
        self.candidate_models = candidate_models

        # Check that all candidate models are of class Model, and that they are all of the same type, pdf or python
        all_notisinstance = [not isinstance(model, Model) for model in self.candidate_models]
        if any(all_notisinstance):
            raise ValueError('All candidate models should be of type Model.')

        all_typesnotequal = [model.type != candidate_models[0].type for model in candidate_models]
        if any(all_typesnotequal):
            raise ValueError('All candidate models should be of same type, pdf of python.')

        # First evaluate ML estimate for all models
        list_models = candidate_models
        list_params = []
        list_criteria = []
        list_penalty_term = []
        for i, model in enumerate(candidate_models):
            with suppress_stdout():
                ml_estimator = MLEstimation(model_instance = model, data = self.data)
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

    def __init__(self, data=None, model=None, pdf_proposal_type=None, pdf_proposal_scale=None,
                 algorithm=None, jump=None, nsamples=None, nburn=None, walkers=None, seed=None):

        if not isinstance(model, Model):
            raise ValueError('model should be of type Model')
        self.model = model
        self.data = data
        self.seed = seed

        # Properties for the MCMC
        if pdf_proposal_type is None:
            self.pdf_proposal_type = 'Uniform'
        else:
            self.pdf_proposal_type = pdf_proposal_type

        if algorithm is None:
            self.algorithm = 'Stretch'
        else:
            self.algorithm = algorithm

        if pdf_proposal_scale is None:
            if self.algorithm == 'Stretch':
                self.pdf_proposal_scale = 2
                if walkers is None:
                    self.walkers = 50
                else:
                    self.walkers = walkers
            else:
                self.pdf_proposal_scale = 1
        else:
            self.pdf_proposal_scale = pdf_proposal_scale

        if nsamples is None:
            self.nsamples = 10000
        else:
            self.nsamples = nsamples

        if jump is None:
            self.jump = 0
        else:
            self.jump = jump

        if nburn is None:
            self.nburn = 0
        else:
            self.nburn = nburn

        self.samples = self.run_bayes_parameter_estimation()
        del self.algorithm, self.jump, self.nburn, self.nsamples, self. pdf_proposal_scale
        del self.pdf_proposal_type

    def run_bayes_parameter_estimation(self):

        if self.model.name is not None:
            print('UQpy: Running parameter estimation for candidate model:', self.model.name)
        else:
            print('UQpy: Running parameter estimation for candidate model')

        if self.seed is None:
            mle = MLEstimation(model_instance=self.model, data=self.data)
            self.seed = mle.param

        self.pdf_target = partial(self.target_post)
        z = MCMC(dimension=self.model.n_params, pdf_proposal_type=self.pdf_proposal_type,
                 pdf_proposal_scale=self.pdf_proposal_scale,
                 algorithm=self.algorithm, jump=self.jump, seed=self.seed, nburn=self.nburn, nsamples=self.nsamples,
                 pdf_target_type='joint_pdf', pdf_target=self.pdf_target)
        
        print('UQpy: Parameter estimation analysis completed!')

        return z.samples

    def target_post(self, theta, args):
        _ = args
        param = np.array(theta)
        '''non-informative prior, p(theta)=1 everywhere'''
        return np.exp(self.model.log_like(self.data, param))


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
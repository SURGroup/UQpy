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
import matplotlib.pyplot as plt


########################################################################################################################
########################################################################################################################
#                            Information theoretic model selection - AIC, BIC
########################################################################################################################

class Model:
    def __init__(self, model_type = None, model_script = None, model_name = None,
                 n_params = None, error_covariance = 1, cpu_RunModel = 1,
                 prior_name = None, prior_params=None, prior_type=None):

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
            self.pdf = Distribution(self.name)
            self.n_params = self.pdf.n_params
            self.log_like = self.log_like_sum

        self.prior_type = prior_type
        if prior_name is not None:
            # first case: define the prior as a list of marginals
            if type(prior_name) is list:
                self.prior = DistributionFromMarginals(name=prior_name, parameters=prior_params)
            # last case: define the prior as a joint pdf
            else:
                self.prior = Distribution(name = prior_name, parameters = prior_params)
        else:
            self.prior = None

    def log_like_normal(self, x, params):
        if np.size(params) == self.n_params:
            params = params.reshape((1, self.n_params))
        if params.shape[1] != self.n_params:
            raise ValueError('the nb of columns in params should be equal to model.n_params')
        with suppress_stdout():  # disable printing output comments
            z = RunModel(cpu=self.cpu_RunModel, model_type=self.type, model_script=self.script, dimension=self.n_params,
                         samples=params)
        results = np.empty((params.shape[0], ), dtype=float)
        for i in range(params.shape[0]):
            mean = z.model_eval.QOI[i]
            results[i] = multivariate_normal.logpdf(x, mean=mean, cov=self.error_covariance)
        return results

    def log_like_sum(self, x, params):
        if np.size(params) == self.n_params:
            params = params.reshape((1, self.n_params))
        if params.shape[1] != self.n_params:
            raise ValueError('the nb of columns in params should be equal to model.n_params')
        results = np.empty((params.shape[0], ), dtype=float)
        for i in range(params.shape[0]):
            param_i = params[i,:].reshape((self.n_params, ))
            results[i] = np.sum(self.pdf.log_pdf(x, param_i))
        return results


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
                print(x0.shape)
                res = minimize(log_like_data, x0, method=method_optim,options = {'disp': True})
                list_param.append(res.x)
                list_max_log_like.append((-1)*res.fun)
            idx_max = int(np.argmax(list_max_log_like))
            self.param = np.array(list_param[idx_max])
            self.max_log_like = list_max_log_like[idx_max]

        elif model_instance.type == 'pdf':
            print('Evaluating max likelihood estimate...')
            self.param = np.array(model_instance.pdf.fit(self.data))
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

    def __init__(self, data=None, model=None, sampling_method = None,
                 pdf_proposal = None, pdf_proposal_type=None, pdf_proposal_scale=None, pdf_proposal_params = None,
                 algorithm=None, jump=None, nsamples=None, nburn=None, walkers=None, seed=None):

        if not isinstance(model, Model):
            raise ValueError('model should be of type Model')
        if data is None:
            raise ValueError('data should be provided')
        if nsamples is None:
            raise ValueError('nsamples should be defined')
        self.nsamples = nsamples
        self.model = model
        self.data = data
        self.sampling_method = sampling_method

        if self.sampling_method == 'MCMC':
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

            if jump is None:
                self.jump = 0
            else:
                self.jump = jump

            if nburn is None:
                self.nburn = 0
            else:
                self.nburn = nburn

            self.samples, self.accept_ratio = self.run_bayes_parameter_estimation()

            del self.algorithm, self.jump, self.nburn, self.nsamples, self. pdf_proposal_scale
            del self.pdf_proposal_type

        elif self.sampling_method == 'IS':

            self.pdf_proposal_params = pdf_proposal_params
            self.pdf_proposal = pdf_proposal
            self.pdf_proposal_type = pdf_proposal_type
            # importance distribution is given by the user
            if self.pdf_proposal is None:
                if self.model.prior is None:
                    raise ValueError('a proposal density or a prior should be given')
                self.pdf_proposal = self.model.prior.name
                self.pdf_proposal_params = self.model.prior.params
                self.pdf_proposal_type = self.model.prior_type

            self.samples, self.weights = self.run_bayes_parameter_estimation()

            del self.nsamples, self.pdf_proposal, self.pdf_proposal_params, self.pdf_proposal_type

        else:
            raise ValueError('sampling_method should be either "MCMC" or "IS"')

    def run_bayes_parameter_estimation(self):

        if self.model.name is not None:
            print('UQpy: Running parameter estimation for candidate model:', self.model.name)
        else:
            print('UQpy: Running parameter estimation for candidate model')

        if self.sampling_method == 'MCMC':
            if self.seed is None:
                mle = MLEstimation(model_instance=self.model, data=self.data)
                self.seed = mle.param

            self.pdf_target = partial(self.target_post)
            self.log_pdf_target = partial(self.log_target_post)
            z = MCMC(dimension=self.model.n_params, pdf_proposal_type=self.pdf_proposal_type,
                     pdf_proposal_scale=self.pdf_proposal_scale,
                     algorithm=self.algorithm, jump=self.jump, seed=self.seed, nburn=self.nburn, nsamples=self.nsamples,
                     pdf_target_type='joint_pdf', pdf_target=self.pdf_target, log_pdf_target=self.log_pdf_target)

            print('UQpy: Parameter estimation analysis completed!')

            return z.samples, z.accept_ratio

        elif self.sampling_method == 'IS':

            self.log_pdf_target = partial(self.log_target_post)

            z = IS(dimension=self.model.n_params, nsamples=self.nsamples,
                   pdf_proposal = self.pdf_proposal, pdf_proposal_params = self.pdf_proposal_params,
                   pdf_proposal_type = self.pdf_proposal_type,
                   pdf_target_type = 'joint_pdf', log_pdf_target=self.log_pdf_target)
            print(z)

            print('UQpy: Parameter estimation analysis completed!')

            return z.samples, z.weights

        else:
            raise ValueError('Only IS and MCMC are supported for inference.')



    def target_post(self, theta, args):
        _ = args
        if type(theta) is not np.ndarray:
            theta = np.array(theta)
        # non-informative prior, p(theta)=1 everywhere
        if self.model.prior is None:
            return np.exp(self.model.log_like(x=self.data, params=theta))
        # prior is given
        else:
            return np.exp(self.model.log_like(x=self.data, params=theta) +
                          self.model.prior.log_pdf(x=theta, params=self.model.prior.params))

    def log_target_post(self, theta, args):
        _ = args
        if type(theta) is not np.ndarray:
            theta = np.array(theta)
        # non-informative prior, p(theta)=1 everywhere
        if self.model.prior is None:
            return self.model.log_like(x=self.data, params=theta)
        # prior is given
        else:
            return self.model.log_like(x=self.data, params=theta) + \
                   self.model.prior.log_pdf(x=theta, params=self.model.prior.params)


class Diagnostics():

    def __init__(self, sampling_method, sampling_outputs):

        if sampling_method not in ['MCMC', 'IS']:
            raise ValueError('Supported sampling methods for diagnostics are "MCMC", "IS".')
        self.sampling_method = sampling_method
        if sampling_outputs is None:
            raise ValueError('Outputs of the sampling procedure should be provided in sampling_outputs.')

        if self.sampling_method == 'IS':
            self.effective_sample_size = 1/np.sum(sampling_outputs.weights**2, axis=0)
            print('Effective sample size is ne={}, for a total number of samples={}'.
                  format(self.effective_sample_size,np.size(sampling_outputs.weights)))
            print('max_weight = {}, min_weight = {}'.format(max(sampling_outputs.weights),
                                                            min(sampling_outputs.weights)))
            # would also be nice to visualize the weights
            fig, ax = plt.subplots()
            ax.hist(sampling_outputs.weights)
            ax.set_title('histogram of the normalized weights from IS')
            plt.show()
            fig, ax = plt.subplots()
            ax.scatter(sampling_outputs.weights, np.zeros((np.size(sampling_outputs.weights), )),
                       s=sampling_outputs.weights*200)
            ax.set_xlabel('weights')
            plt.show()

        if self.sampling_method == 'MCMC':

            samples = sampling_outputs.samples
            nsamples, nparams = samples.shape

            # Acceptance ratio
            print('Acceptance ratio of the chain = {}'.format(sampling_outputs.accept_ratio))

            # Computation of ESS and min ESS
            eps = 0.05
            alpha = 0.05

            bn = np.ceil(nsamples**(1/2))
            an = int(np.ceil(nsamples/bn))
            idx = np.array_split(np.arange(nsamples), an)

            means_subdivisions = np.empty((an, samples.shape[1]))
            for i, idx_i in enumerate(idx):
                x_sub = samples[idx_i, :]
                means_subdivisions[i,:] = np.mean(x_sub, axis=0)
            Omega = np.cov(samples.T)
            Sigma = np.cov(means_subdivisions.T)
            joint_ESS = nsamples*np.linalg.det(Omega)**(1/nparams)/np.linalg.det(Sigma)**(1/nparams)
            chi2_value = chi2.ppf(1 - alpha, df=nparams)
            min_joint_ESS = 2 ** (2 / nparams) * np.pi / (nparams * gamma(nparams / 2)) ** (
                        2 / nparams) * chi2_value / eps ** 2
            marginal_ESS = np.empty((nparams, ))
            min_marginal_ESS = np.empty((nparams,))
            for j in range(nparams):
                marginal_ESS[j] = nsamples * Omega[j,j]/Sigma[j,j]
                min_marginal_ESS[j] = 4 * norm.ppf(alpha/2)**2 / eps**2

            print('Multivariate ESS = {}, minESS = {}'.format(joint_ESS, min_joint_ESS))
            print('Univariate ESS in each dimension')
            for j in range(nparams):
                print('Parameter {}: ESS = {}, minESS = {}'.format(j+1, marginal_ESS[j], min_marginal_ESS[j]))

            # Outputs plots
            fix, ax = plt.subplots(nrows=nparams, ncols=3, figsize=(10,10))
            for j in range(samples.shape[1]):
                ax[j,0].plot(np.arange(nsamples), samples[:,j])
                ax[j,0].set_title('chain for parameter # {}'.format(j+1))
                ax[j,1].plot(np.arange(nsamples), np.cumsum(samples[:,j])/np.arange(nsamples))
                ax[j,1].set_title('parameter convergence')
                ax[j,2].acorr(samples[:,j]-np.mean(samples[:,j]), maxlags = 50, normed=True)
                ax[j,2].set_title('ESS = {}'.format(marginal_ESS[j]))
            plt.show()

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
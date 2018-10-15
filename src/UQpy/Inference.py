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
import warnings

warnings.filterwarnings("ignore")


########################################################################################################################
########################################################################################################################
#                            Information theoretic model selection - AIC, BIC
########################################################################################################################

class InfoModelSelection:

    def __init__(self, data=None, candidate_model=None, method=None):

        """

        :param data: Available data
        :type data: ndarray

        :param candidate_model: Candidate non-parametric probabilistic models
        :type candidate_model: list

        :param method: Method to be used
        :type method: str

        """

        self.data = data
        self.method = method
        self.candidate_model = list()

        for i in range(len(candidate_model)):
            self.candidate_model.append(Distribution(candidate_model[i]))

        if self.method == 'AIC':
            self.AIC, self.probabilities, self.delta, self.sorted_model, self.Parameters = self.run_multi_info_ms()
        elif self.method == 'AICc' or self.method is None:
            self.AICc, self.probabilities, self.delta, self.sorted_model, self.Parameters = self.run_multi_info_ms()
        elif self.method == 'BIC':
            self.BIC, self.probabilities, self.delta, self.sorted_model, self.Parameters = self.run_multi_info_ms()

    def run_multi_info_ms(self):

        criterion = np.zeros((len(self.candidate_model)))
        model_sort = ["" for x in range(len(self.candidate_model))]
        params = [0] * len(self.candidate_model)
        for i in range(len(self.candidate_model)):
            print('UQpy: Running Informative model selection for candidate model:', self.candidate_model[i].name)
            criterion[i], params[i] = self.info_ms(self.candidate_model[i])

        criterion_sort = np.sort(criterion)
        params_sort = [0] * len(self.candidate_model)
        sort_index = sorted(range(len(self.candidate_model)), key=criterion.__getitem__)
        for i in range((len(self.candidate_model))):
            s = sort_index[i]
            model_sort[i] = self.candidate_model[s].name
            params_sort[i] = params[s]

        criterion_delta = criterion_sort - np.nanmin(criterion_sort)

        s_ = 1.0
        w_criterion = np.empty([len(self.candidate_model), 1], dtype=np.float16)
        w_criterion[0] = 1.0

        delta = np.empty([len(self.candidate_model), 1], dtype=np.float16)
        delta[0] = 0.0

        for i in range(1, len(self.candidate_model)):
            delta[i] = criterion_sort[i] - criterion_sort[0]
            w_criterion[i] = np.exp(-delta[i] / 2)
            s_ = s_ + w_criterion[i]

        probabilities = w_criterion / s_
        print('UQpy: Informative model selection analysis completed!')

        return criterion_sort, probabilities, criterion_delta, model_sort, params_sort

    def info_ms(self, model):

        n = self.data.shape[0]
        fit_i = model.fit
        params = list(fit_i(self.data))
        log_like = np.sum(model.log_pdf(self.data, params))
        k = model.n_params
        aic_value = 2 * n - 2 * log_like
        aicc_value = 2 * n - 2 * log_like + (2 * k ** 2 + 2 * k) / (n - k - 1)
        bic_value = np.log(n) * k - 2 * log_like

        if self.method == 'BIC':
            return bic_value, params
        elif self.method == 'AIC':
            return aic_value, params
        elif self.method == 'AICc' or self.method is None:
            return aicc_value, params


########################################################################################################################
########################################################################################################################
#                                  Bayesian Parameter estimation
########################################################################################################################
class BayesParameterEstimation:

    def __init__(self, data=None, model=None, theta_prior_params=None, method=None, theta_prior_dist=None,
                 pdf_proposal_type=None, pdf_proposal_scale=None, algorithm=None, jump=None, nsamples=None, nburn=None,
                 walkers=None):

        self.data = data
        self.method = method

        if model is None:
            raise RuntimeError('A probability model is required.')
        elif isinstance(model, str):
            self.model = Distribution(model)
        elif isinstance(model, list):
            self.model = Distribution(model[0])
        elif isinstance(model, Distribution) is True:
            self.model = model

        if theta_prior_dist is None and theta_prior_params is None:
            self.theta_prior_distribution = Distribution('Uniform', [0, 10 ** 6])

        elif theta_prior_params is None and theta_prior_dist is not None:
            raise RuntimeError('The parameters of the prior distribution models should be provided.')

        elif theta_prior_dist is not None and theta_prior_params is not None:
            if isinstance(theta_prior_dist, str) is True and isinstance(theta_prior_params, str) is True:
                self.theta_prior_distribution = Distribution(theta_prior_dist[0], theta_prior_params)
            elif isinstance(theta_prior_dist, list):
                self.theta_prior_distribution = Distribution(theta_prior_dist[0], theta_prior_params)
            elif isinstance(theta_prior_dist, str) is True:
                self.theta_prior_distribution = Distribution(theta_prior_dist, theta_prior_params)
            elif isinstance(theta_prior_dist, list) is True and isinstance(theta_prior_dist, list) is True:
                self.theta_prior_distribution = Distribution(theta_prior_dist[0], theta_prior_params[0])
            elif isinstance(theta_prior_dist, Distribution) is True:
                self.theta_prior_distribution = theta_prior_dist
            else:
                print('Error: The prior models should be of type string or of type Distribution.')

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
            self.jump = 1
        else:
            self.jump = jump

        if nburn is None:
            self.nburn = 1
        else:
            self.nburn = nburn

        self.samples, self.MLE = self.run_bayes_parameter_estimation()
        del self.algorithm, self.jump, self.method, self.nburn, self.nsamples, self.pdf_proposal_scale
        del self.pdf_proposal_type

    def run_bayes_parameter_estimation(self):
        print('UQpy: Running parameter estimation for candidate model:', self.model.name)

        if self.model.n_params > 3:
            raise RuntimeError('Only distributions with up to three-dimensional parameters are available in UQpy.')

        from UQpy.SampleMethods import MCMC

        seed = np.array(self.model.fit(self.data))

        '''
        z = MCMC(dimension=self.model.n_params, pdf_proposal_type=self.pdf_proposal_type,
                 pdf_proposal_scale=self.pdf_proposal_scale,
                 algorithm=self.algorithm, jump=self.jump, seed=seed, nburn=self.nburn, nsamples=self.nsamples,
                 pdf_target_type='joint_pdf', pdf_target=self.ln_prob)

        samples = z.samples
        '''

        import emcee

        p0 = [seed + 1e-3 * np.random.rand(self.model.n_params) for i in range(self.walkers)]
        sampler = emcee.EnsembleSampler(self.walkers, self.model.n_params, self.lnprob, args=[self.data])
        sampler.run_mcmc(p0, self.nsamples)
        samples = sampler.chain[:, 50:, :].reshape((-1, self.model.n_params))

        log_like = np.zeros(samples.shape[0])
        for i in range(samples.shape[0]):
            log_like[i] = np.sum(self.model.log_pdf(self.data, samples[i, :]))

        print('UQpy: Parameter estimation analysis completed!')

        if self.method == 'MLE' or self.method is None:
            return samples, samples[np.argmax(log_like)]

    def lnprob(self, theta, args):
        _ = args
        '''non-informative prior'''
        lp = non_info_prior_dist(self.model.name, theta)

        if not np.isfinite(lp):
            return -np.inf

        a = self.model.log_pdf(self.data, theta)
        loglike = np.sum(a)

        return (lp + loglike)


########################################################################################################################
########################################################################################################################
#                                  Bayesian Inference
########################################################################################################################

class BayesModelSelection:

    def __init__(self, data=None, dimension=None, candidate_model=None, pdf_proposal_type=None, pdf_proposal_scale=None,
                 algorithm=None, jump=None, nsamples=None, nburn=None, prior_probabilities=None,
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
                self.prior_hyperparams[i] = [0, 10 ** 6]

        if prior_hyperparams is None and param_prior_dist is not None:
            raise RuntimeError('The parameters of the prior distribution models should be provided.')

        if prior_hyperparams is not None:
            self.prior_hyperparams = [0] * len(prior_hyperparams)
            for i in range(len(prior_hyperparams)):
                self.prior_hyperparams[i] = prior_hyperparams[i]

        if param_prior_dist is not None:
            self.param_prior_dist = [0] * len(param_prior_dist)
            for i in range(len(param_prior_dist)):
                self.param_prior_dist[i] = param_prior_dist[i]

        if method is None:
            self.method = 'MLE'
        else:
            self.method = method

        if prior_probabilities is None:
            self.prior_probabilities = [0] * len(self.candidate_model)
            for i in range(len(self.candidate_model)):
                self.prior_probabilities[i] = 1 / len(self.candidate_model)
        else:
            self.prior_probabilities = prior_probabilities

        self.model_probabilities, self.evidence, self.parameter_estimation = self.run_multi_bayes_ms()

    def run_multi_bayes_ms(self):
        # Initialize the evidence or marginal likelihood
        evi_value = np.zeros(len(self.candidate_model))
        scaled_evi = np.zeros_like(evi_value)
        pe = [0] * len(self.candidate_model)
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
            scaled_evi[i] = evi_value[i] * self.prior_probabilities[i]

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

        if params[0] < 0.0028 or params[2] < 0:
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
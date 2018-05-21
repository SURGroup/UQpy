from UQpy import SampleMethods, PDFs
import scipy.stats as stats
import numpy as np
import emcee
from scipy import integrate


def init_inf(data):
    ################################################################################################################
    # Add available sampling methods Here
    valid_methods = ['InforMS', 'MultiMI', 'Bayes_Inference', 'MultiMBayesI']

    ################################################################################################################
    # Check if requested method is available
    if 'Method' in data:
        if data['Method'] not in valid_methods:
            raise NotImplementedError("Method - %s not available" % data['Method'])
    else:
        raise NotImplementedError("No Bayesian Inference method was provided")

    ####################################################################################################################
    # TODO: Jiaxin: InforMS block.
    # Necessary  Information theoretic model selection - AIC, BIC parameters:
    # Optional:

    ####################################################################################################################
    # TODO: Jiaxin: MultiMI block.
    # Necessary Multimodel Information-theoretic selection parameters:
    # Optional:


    ####################################################################################################################
    # TODO: Jiaxin: Bayes_Inference block.
    # Necessary  Bayesian inference - parameter estimation parameters:
    # Optional:


    ####################################################################################################################
    # TODO: Jiaxin: MultiMBayesI block.
    # Necessary  Multimodel Bayesian inference parameters:
    # Optional:


# This is a new class - inference
# Created by: Jiaxin Zhang (As one of authors or contributors of UQpy?)
# Author(Contributor???): Jiaxin Zhang
# Last modified by Dimitris Giovanis: 4/2/2018

# TODO: 16/56 todo list!
# 1. Information theoretic model selection √
# 2. Information theoretic multimodel selection √
# 3. Bayesian parameter estimation - Multimodel
# 4. Bayesian model selection
# 5. Bayesian parameter estimation - Conventional MLE
# 6. Optimal sampling density
# 7. Copula model selection
# 8. Copula multimodel selection
# 9. Copula parameter selection
# 10. Multimodel kriging??

# 11. Global Sensitivity Analysis (sampling class)
# 12. Importance sampling (sampling class)
# 13. Partially Stratified Sampling (sampling class)
# 14. Latinized Stratifed Sampling (sampling class)
# 15. Latinized Partially Stratified Sampling (sampling class)
# 16. Optimal sampling density (sampling or inference class)

# TODO：01-15-2018
# 1. using sampleMethods MCMC class or Ensemble MCMC
# 3. Bayesian prior information input
# 4. Bayesian model selection debug
# 5. Multimodel information theoretic selection
# 6. Bayesian parameter estimation - Conventional MLE

########################################################################################################################
#                                        Information theoretic model selection - AIC, BIC
########################################################################################################################


class InforMS:

    def __init__(self, data=None, model=None):
        """
        Created by: Jiaxin Zhang
        Last modified by Dimitris Giovanis: 4/2/2018
        """
        # TODO: more candidate probability models:
        # normal, lognormal, gamma, inverse gaussian, logistic, cauchy, exponential
        # weibull, loglogistic ??

        n = len(data)

        if model == 'normal':
            fitted_params_norm = stats.norm.fit(data)
            k, loglike = PDFs.log_normal(data, fitted_params_norm)

        elif model == 'cauchy':
            fitted_params_cauchy = stats.cauchy.fit(data)
            k, loglike = PDFs.log_cauchy(data, fitted_params_cauchy)

        elif model == 'exponential':
            fitted_params_expon = stats.expon.fit(data)
            k, loglike = PDFs.log_exp(data, fitted_params_expon)

        elif model == 'lognormal':
            fitted_params_logn = stats.lognorm.fit(data)
            k, loglike = PDFs.log_log(data, fitted_params_logn)

        elif model == 'gamma':
            fitted_params_gamma = stats.gamma.fit(data)
            k, loglike = PDFs.log_gamma(data, fitted_params_gamma)

        elif model == 'invgauss':
            fitted_params_invgauss = stats.invgauss.fit(data)
            k, loglike = PDFs.log_invgauss(data, fitted_params_invgauss)

        elif model == 'logistic':
            fitted_params_logistic = stats.logistic.fit(data)
            k, loglike = PDFs.log_logistic(data, fitted_params_logistic)

        aic_value = 2 * n - 2 * (loglike)
        aicc_value = 2 * n - 2 * (loglike) + (2 * k ** 2 + 2 * k) / (n - k - 1)
        bic_value = np.log(n) * k - 2 * (loglike)

        self.AIC = aic_value
        self.AICC = aicc_value
        self.BIC = bic_value

    def init_InforrMS(self):
        print()
        # TODO: ADD CHECKS for Information theoretic model selection - AIC, BIC


########################################################################################################################
#                                        Multimodel Information-theoretic selection
########################################################################################################################


class MultiMI:

    """
    Created by: Jiaxin Zhang
    Last modified by Dimitris Giovanis: 4/2/2018
    """

    def __init__(self, data=None, model=None):

        value = np.zeros((len(model)))
        model_sort = ["" for x in range(len(model))]

        for i in range(len(model)):
            model0 = model[i]
            value[i] = InforMS(data=data, model=model0).AICC

        v_sort = np.sort(value)
        sort_index = sorted(range(len(value)), key=value.__getitem__)
        for i in range((len(value))):
            s = sort_index[i]
            model_sort[i] = model[s]

        v_delta = v_sort - np.min(v_sort)

        s_AIC = 1.0
        w_AIC = np.empty([len(model), 1], dtype=np.float16)
        w_AIC[0] = 1.0

        delta = np.empty([len(model), 1], dtype=np.float16)
        delta[0] = 0.0

        for i in range(1, len(model)):
            delta[i] = v_sort[i] - v_sort[0]
            w_AIC[i] = np.exp(-delta[i] / 2)
            s_AIC = s_AIC + w_AIC[i]

        weights = w_AIC / s_AIC

        self.AICC = v_sort
        self.weights = weights
        self.delta = v_delta
        self.model_sort = model_sort

    def init_MultiMI(self):
        print()
        # TODO: Jiaxin: Multimodel Information-theoretic selection

########################################################################################################################
#                                         Bayesian inference - parameter estimation
########################################################################################################################


class Bayes_Inference:
    """
    Created by: Jiaxin Zhang
    Last modified by Dimitris Giovanis: 4/2/2018
    """

    # TODO: prior type - noninformative, informative
    # TODO: prior distribution type - uniform, beta, normal etc.
    # TODO: MCMC algorithms - emcee

    def __init__(self, data=None, model=None):

        def lnlike(theta, data=data):
            if model == 'normal':
                k, loglike = PDFs.log_normal(data, theta)

            elif model == 'cauchy':
                k, loglike = PDFs.log_cauchy(data, theta)

            elif model == 'exponential':
                k, loglike = PDFs.log_exp(data, theta)

            elif model == 'lognormal':
                k, loglike = PDFs.log_log(data, theta)

            elif model == 'gamma':
                k, loglike = PDFs.log_gamma(data, theta)

            elif model == 'invgauss':
                k, loglike = PDFs.log_invgauss(data, theta)

            elif model == 'logistic':
                k, loglike = PDFs.log_logistic(data, theta)

            return loglike

        def lnprior(theta):
            m, s = theta
            if -10000.0 < m < 10000.0 and 0.0 < s < 10000.0:
                return 0.0
            return -np.inf

        def lnprob(theta, data=data):
            lp = lnprior(theta)
            if not np.isfinite(lp):
                return -np.inf

            #return np.exp(lp + lnlike(theta, data))
            return (lp + lnlike(theta, data))

        # TODO: computing the evidence using numerical integration; monte carlo; RJMCMC?

        def integrate_posterior_2D(lnprob, xlim, ylim, data=data):
            func = lambda theta1, theta0: np.exp(lnprob([theta0, theta1], data))
            return integrate.dblquad(func, xlim[0], xlim[1], lambda x: ylim[0], lambda x: ylim[1])

        nwalkers = 50
        ndim = 2
        p0 = [np.random.rand(ndim) for i in range(nwalkers)]

        #target_like = np.exp(lnprob)

        # # using MCMC from SampleMethods
        #MCMC = SampleMethods.MCMC(nsamples = 5000, dimension = 2, seed = [2,1], algorithm = 'MH',
        #                          pdf_proposal_type = 'Normal',
        #                          pdf_proposal_width = 1, pdf_target_type = lnprob, skip = 1,
        #                          pdf_target_params = [[0, 1], [0, 1]])

        #print(MCMC.samples)
        #plt.plot(MCMC.samples[:,0], MCMC.samples[:,1], 'ro')
        #plt.show()

        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[data])
        sampler.run_mcmc
        trace = sampler.chain[:, 50:, :].reshape((-1, ndim))

        #fig = corner.corner(trace, labels=["$m$", "$s$"], truths=[0, 1, np.log(1)])
        #fig.show()
        #fig.savefig("pos_samples.png")
        # plt.plot(trace[:, 0], trace[:, 1], 'ko')
        # plt.show()

        #Bayesian parameter estimation - Conventional MLE
        # print(trace)
        # print(trace[0])
        loglike = np.zeros((len(trace)))
        for j in range(len(trace)):
            loglike[j] = lnlike(theta=trace[j], data=data)

        index = np.argmax(loglike)
        print(model)
        mle_Bayes = trace[index]
        print('MLE_Bayes:', mle_Bayes)

        # Bayes factor
        xlim, ylim = zip(trace.min(0), trace.max(0))

        Z1, err_Z1 = integrate_posterior_2D(lnprob, xlim, ylim)
        #print("Z1 =", Z1, "+/-", err_Z1)

        self.samples = trace
        self.Bayes_factor = Z1
        self.Bayes_mle = mle_Bayes

    def init_Bayes_Inference(self):
        print()
        # TODO: Jiaxin: ADD CHECKS for Bayesian inference - parameter estimation

########################################################################################################################
#                                  Multimodel Bayesian inference
########################################################################################################################


class MultiMBayesI:
    """
    Created by: Jiaxin Zhang
    Last modified by Dimitris Giovanis: 4/2/2018
    """

    def __init__(self, data=None, model=None):
        evi_value = np.zeros(len(model))
        mle_value = np.zeros(len(model))
        model_sort = ["" for x in range(len(model))]
        for i in range(len(model)):
            model0 = model[i]
            evi_value[i] = Bayes_Inference(data=data, model=model0).Bayes_factor
            # mle_value[i, :] = Inference.Bayes_Inference(data=data, model=model0).Bayes_mle

        sum_evi_value = np.sum(evi_value)
        nevi_value = -evi_value

        sort_index = sorted(range(len(nevi_value)), key=nevi_value.__getitem__)
        for i in range((len(nevi_value))):
            s = sort_index[i]
            model_sort[i] = model[s]

        bms = -np.sort(nevi_value)/sum_evi_value

        self.model_sort = model_sort
        self.weights = bms
        # self.mle_Bayes = mle_value

    def init_MultiMBayesI(self):
        print()
        # TODO: Jiaxin: ADD CHECKS for Multimodel Bayesian inference
import scipy.stats as stats
from scipy.special import erf
from functools import partial
import numpy as np
import sys
import os


########################################################################################################################
#        Define the probability distribution of the random parameters
########################################################################################################################

def pdf(dist):
    dir_ = os.getcwd()
    sys.path.insert(0, dir_)
    import custom_pdf
    method_to_call = getattr(custom_pdf, dist)

    return partial(method_to_call)


# TODO: Add a library of pdfs here.

########################################################################################################################
#        Transform the random parameters from U(0, 1) to the original space
########################################################################################################################

def inv_cdf(x, pdf_type, params):
    x_trans = np.zeros(shape=(x.shape[0], x.shape[1]))
    ###################################################################################
    # U(0, 1)  ---->  U(a, b)

    for i in range(x.shape[1]):
        if pdf_type[i] == 'Uniform':
            for j in range(x.shape[0]):
                x_trans[j, i] = stats.uniform.ppf(x[j, i], params[i][0], params[i][1])

        ###################################################################################
        # U(0, 1)  ---->  N(μ, σ)

        elif pdf_type[i] == 'Normal':
            for j in range(x.shape[0]):
                x_trans[j, i] = stats.norm.ppf(x[j, i], params[i][0], params[i][1])

        ####################################################################################
        # U(0, 1)  ---->  LN(μ, σ)

        elif pdf_type[i] == 'Lognormal':
            for j in range(x.shape[0]):
                x_trans[j, i] = stats.lognorm.ppf(x[j, i], params[i][0], params[i][1])

        ####################################################################################
        # U(0, 1)  ---->  Weibull(λ, κ)

        elif pdf_type[i] == 'Weibull':
            for j in range(x.shape[0]):
                x_trans[j, i] = stats.weibull_min.ppf(x[j, i], params[i][0], params[i][1])

        ####################################################################################
        # U(0, 1)  ---->  Beta(q, r, a, b)

        elif pdf_type[i] == 'Beta':
            for j in range(x.shape[0]):
                x_trans[j, i] = stats.beta.ppf(x[j, i], params[i][0], params[i][1], params[i][2], params[i][3])

        ####################################################################################
        # U(0, 1)  ---->  Exp(λ)

        elif pdf_type[i] == 'Exponential':
            for j in range(x.shape[0]):
                x_trans[j, i] = stats.expon.ppf(x[j, i], params[i][0])

        ####################################################################################
        # U(0, 1)  ---->  Gamma(λ-shape, shift, scale )

        elif pdf_type[i] == 'Gamma':
            for j in range(x.shape[0]):
                x_trans[j, i] = stats.gamma.ppf(x[j, i], params[i][0], params[i][1], params[i][2])

    return x_trans


########################################################################################################################
#        Get probability of a pdf
########################################################################################################################

def prob_pdf(x, pdf_type, params):
    prob = np.zeros(shape=(x.shape[0], x.shape[1]))

    for i in range(x.shape[1]):
        if pdf_type[i] == 'Uniform':
            for j in range(x.shape[0]):
                prob[j, i] = stats.uniform.pdf(x[j, i], params[i][0], params[i][1])

        elif pdf_type[i] == 'Normal':
            for j in range(x.shape[0]):
                prob[j, i] = stats.norm.pdf(x[j, i], params[i][0], params[i][1])

        elif pdf_type[i] == 'Lognormal':
            for j in range(x.shape[0]):
                prob[j, i] = stats.lognorm.pdf(x[j, i], params[i][0], params[i][1])

        elif pdf_type[i] == 'Weibull':
            for j in range(x.shape[0]):
                prob[j, i] = stats.weibull_min.pdf(x[j, i], params[i][0], params[i][1])

        elif pdf_type[i] == 'Beta':
            for j in range(x.shape[0]):
                prob[j, i] = stats.beta.pdf(x[j, i], params[i][0], params[i][1], params[i][2], params[i][3])

        elif pdf_type[i] == 'Exponential':
            for j in range(x.shape[0]):
                prob[j, i] = stats.expon.pdf(x[j, i], params[i][0])

        elif pdf_type[i] == 'Gamma':
            for j in range(x.shape[0]):
                prob[j, i] = stats.gamma.pdf(x[j, i], params[i][0], params[i][1], params[i][2])

    return prob


########################################################################################################################
#        Get cumulative probability of a pdf
########################################################################################################################

def prob_cdf(x, pdf_type, params):
    prob_c = np.zeros(shape=(x.shape[0], x.shape[1]))

    for i in range(x.shape[1]):
        if pdf_type[i] == 'Uniform':
            for j in range(x.shape[0]):
                prob_c[j, i] = stats.uniform.cdf(x[j, i], params[i][0], params[i][1])

        elif pdf_type[i] == 'Normal':
            for j in range(x.shape[0]):
                prob_c[j, i] = stats.norm.cdf(x[j, i], params[i][0], params[i][1])

        elif pdf_type[i] == 'Lognormal':
            for j in range(x.shape[0]):
                prob_c[j, i] = stats.lognorm.cdf(x[j, i], params[i][0], params[i][1])

        elif pdf_type[i] == 'Weibull':
            for j in range(x.shape[0]):
                prob_c[j, i] = stats.weibull_min.cdf(x[j, i], params[i][0], params[i][1])

        elif pdf_type[i] == 'Beta':
            for j in range(x.shape[0]):
                prob_c[j, i] = stats.beta.cdf(x[j, i], params[i][0], params[i][1], params[i][2], params[i][3])

        elif pdf_type[i] == 'Exponential':
            for j in range(x.shape[0]):
                prob_c[j, i] = stats.expon.cdf(x[j, i], params[i][0])

        elif pdf_type[i] == 'Gamma':
            for j in range(x.shape[0]):
                prob_c[j, i] = stats.gamma.cdf(x[j, i], params[i][0], params[i][1], params[i][2])

    return prob_c

# ########################################################################################################################
#             Log pdf (used in inference)
# ######################################################################################################################
#
# def log_normal(data, fitted_params_norm):
#     loglike = np.sum(stats.norm.logpdf(data, loc=fitted_params_norm[0], scale=fitted_params_norm[1]))
#     k = 2
#     return k, loglike
#
#
# def log_cauchy(data, fitted_params_cauchy):
#     loglike = np.sum(stats.cauchy.logpdf(data, loc=fitted_params_cauchy[0], scale=fitted_params_cauchy[1]))
#     k = 2
#     return k, loglike
#
#
# def log_exp(data, fitted_params_expon):
#     loglike = np.sum(stats.expon.logpdf(data, loc=fitted_params_expon[0], scale=fitted_params_expon[1]))
#     k = 2
#     return k, loglike
#
#
# def log_log(data, fitted_params_logn):
#     loglike = np.sum(stats.lognorm.logpdf(data, s=fitted_params_logn[0], loc=fitted_params_logn[1],
#                                           scale=fitted_params_logn[2]))
#     k = 3
#     return k, loglike
#
#
# def log_gamma(data, fitted_params_gamma):
#     loglike = np.sum(stats.gamma.logpdf(data, a=fitted_params_gamma[0], loc=fitted_params_gamma[1],
#                                         scale=fitted_params_gamma[2]))
#     k = 3
#     return k, loglike
#
#
# def log_invgauss(data, fitted_params_invgauss):
#     loglike = np.sum(stats.invgauss.logpdf(data, mu=fitted_params_invgauss[0], loc=fitted_params_invgauss[1],
#                                            scale=fitted_params_invgauss[2]))
#     k = 3
#     return k, loglike
#
#
# def log_logistic(data, fitted_params_logistic):
#     loglike = np.sum(
#         stats.logistic.logpdf(data, loc=fitted_params_logistic[0], scale=fitted_params_logistic[1]))
#     k = 2
#     return k, loglike

import scipy.stats as stats
from scipy.special import erf
from functools import partial
import numpy as np
import sys
import os


########################################################################################################################
#        Define the probability distribution of the random parameters
########################################################################################################################

def multivariate_pdf(x, dim):
    """ Multivariate normal density function used to generate samples using the Metropolis-Hastings Algorithm
    :math: `f(x_{1},...,x_{k}) = \\frac{1}{((2*\\pi)^{k}*\\Sigma)^(1/2)}*exp(-\\frac{1}{2}*(x-\\mu)^{T}*\\Sigma^{-1}*(x-\\mu))`

    """
    if dim == 1:
        return stats.norm.pdf(x, 0, 1)
    else:
        return stats.multivariate_normal.pdf(x, mean=np.zeros(dim), cov=np.identity(dim))


def marginal_pdf(x, mp):
    """
    Marginal target density used to generate samples using the Modified Metropolis-Hastings Algorithm

    :math:`f(x) = \\frac{1}{\\sqrt{2\\pi\\sigma}}\\exp{-\\frac{1}{2}{\\frac{x-\\mu}{\\sigma}}^2}`

    """
    return stats.norm.pdf(x, mp[0], mp[1])


def pdf(dist):
    if dist == 'multivariate_pdf':
        return partial(multivariate_pdf)

    if dist == 'Gamma':
        return partial(Gamma)

    if dist == 'marginal_pdf':
        return partial(marginal_pdf)

    if dist not in ['multivariate_pdf','Gamma', 'marginal_pdf']:
        dir_ = os.getcwd()
        sys.path.insert(0, dir_)
        import custom_pdf
        method_to_call = getattr(custom_pdf, dist)

        return partial(method_to_call)


########################################################################################################################
#        Define the cumulative distribution of the random parameters
########################################################################################################################

def Gamma(x, params):
    return stats.gamma.cdf(x, params[0], loc=params[1], scale=params[2])


########################################################################################################################
#        Transform the random parameters from U(0, 1) to the original space
########################################################################################################################

def inv_cdf(x, pdf, params):
    x_trans = np.zeros(shape=(x.shape[0], x.shape[1]))
    ###################################################################################
    # U(0, 1)  ---->  U(a, b)

    for i in range(x.shape[1]):
        if pdf[i] == 'Uniform':
            for j in range(x.shape[0]):
                x_trans[j, i] = ppfUniform(x[j, i], params[i][0], params[i][1])

    ###################################################################################
    # U(0, 1)  ---->  N(μ, σ)

        elif pdf[i] == 'Normal':
            for j in range(x.shape[0]):
                x_trans[j, i] = ppfNormal(x[j, i], params[i][0], params[i][1])

    ####################################################################################
    # U(0, 1)  ---->  LN(μ, σ)

        elif pdf[i] == 'Lognormal':
            for j in range(x.shape[0]):
                x_trans[j, i] = ppfLognormal(x[j, i], params[i][0], params[i][1])

    ####################################################################################
    # U(0, 1)  ---->  Weibull(λ, κ)

        elif pdf[i] == 'Weibull':
            for j in range(x.shape[0]):
                x_trans[j, i] = ppfWeibull(x[j, i], params[i][0], params[i][1])

    ####################################################################################
    # U(0, 1)  ---->  Beta(q, r, a, b)

        elif pdf[i] == 'Beta':
            for j in range(x.shape[0]):
                x_trans[j, i] = ppfBeta(x[j, i], params[i][0], params[i][1], params[i][2], params[i][3])

    ####################################################################################
    # U(0, 1)  ---->  Exp(λ)

        elif pdf[i] == 'Exponential':
            for j in range(x.shape[0]):
                x_trans[j, i] = ppfExponential(x[j, i], params[i][0])

    ####################################################################################
    # U(0, 1)  ---->  Gamma(λ-shape, shift, scale )

        elif pdf[i] == 'Gamma':
            for j in range(x.shape[0]):
                x_trans[j, i] = ppfGamma(x[j, i], params[i][0], params[i][1], params[i][2])

    return x_trans


########################################################################################################################
#             Inverse pdf
# ######################################################################################################################


def ppfNormal(p, mu, sigma):
    """Returns the evaluation of the percent point function (inverse cumulative
    distribution) evaluated at the probability p with mean (mu) and
    scale (sigma)."""
    return stats.norm.ppf(p, loc=mu, scale=sigma)


def ppfLognormal(p, mu, sigma):
    """Returns the evaluation of the percent point function (inverse cumulative
    distribution) evaluated at the probability p with mean (mu) and
    scale (sigma)."""
    epsilon = np.sqrt(np.log((sigma**2 + mu**2)/(mu**2)))
    elamb = mu**2/(np.sqrt(mu**2+sigma**2))
    return stats.lognorm.ppf(p, epsilon, scale=elamb)


def ppfWeibull(p, lamb, k):
    """Returns the evaluation of the percent point function (inverse cumulative
    distribution) evaluated at the probability p with scale (lamb) and
    shape (k) specified for a Weibull distribution."""

    # PDF form of Weibull Distirubtion:
    # f(x) = k/lamb * (x/lamb)**(k-1) * exp(-(x/lamb)**k)

    # frechet_r is analogous to weibull-min, or standard weibull.
    return stats.frechet_r.ppf(p, k, scale=lamb)


def ppfUniform(p, a, b):
    """Returns the evaluation of the percent point function (inverse cumulative
    distribution) evaluated at the probability p for a Uniform distribution
    with range (a,b). Usage:\n ppfUniform(a,b)"""
    return a+p*(b-a)


def ppfTriangular(p, a, c, b):
    """Returns the evaluation of the percent point function (inverse cumulative
    distribution) evaluated at the probability p for a triangular distribution.
    Usage:\n ppfTriangular(p, a, c, b)"""
    width = b-a
    scaledMiddle = (c-a)/width
    return stats.triang.ppf(p, scaledMiddle, loc=a, scale=width)


def ppfBeta(p, q, r, a, b):
    """Returns the evaluation of the percent point function (inverse cumulative
    distribution) evaluated at the probability p for a Beta distribution.
    Usage:\n ppfBeta(p, q, r, a, b)"""
    width = b-a
    return stats.beta.ppf(p, q, r, loc=a, scale=width)


def ppfExponential(p, lamb):
    """Returns the evaluation of the percent point function (inverse cumulative
    distribution) evaluated at the probability p for an Exponential
    distribution.  Usage:\n
    ppfExponential(p, lamb)"""
    scalE = 1.0/lamb
    return stats.expon.ppf(p, scale=scalE)


def ppfGamma(p, shape, shift, scale):
    """Returns the evaluation of the percent point function (inverse cumulative
    distribution) evaluated at the probability p for an Gamma
    distribution.  Usage:\n
    ppfGamma(p, shape, shift, scale)"""
    return stats.gamma.ppf(p, shape, loc=shift, scale=scale)


def normal_to_uniform(u, a, b):
    x = np.zeros(shape=(u.shape[0], u.shape[1]))
    for i in range(u.shape[1]):
        p = 0.5 + erf(((u[:, i] - 0) / 1) / np.sqrt(2)) / 2
        x[:, i] = a + (b - a) * p
    return x

########################################################################################################################
#             Log pdf (used in inference)
# ######################################################################################################################

def log_normal(data, fitted_params_norm):
    loglike = np.sum(stats.norm.logpdf(data, loc=fitted_params_norm[0], scale=fitted_params_norm[1]))
    k = 2
    return k, loglike


def log_cauchy(data, fitted_params_cauchy):
    loglike = np.sum(stats.cauchy.logpdf(data, loc=fitted_params_cauchy[0], scale=fitted_params_cauchy[1]))
    k = 2
    return k, loglike


def log_exp(data, fitted_params_expon):
    loglike = np.sum(stats.expon.logpdf(data, loc=fitted_params_expon[0], scale=fitted_params_expon[1]))
    k = 2
    return k, loglike


def log_log(data, fitted_params_logn):
    loglike = np.sum(stats.lognorm.logpdf(data, s=fitted_params_logn[0], loc=fitted_params_logn[1],
                                          scale=fitted_params_logn[2]))
    k = 3
    return k, loglike


def log_gamma(data, fitted_params_gamma):
    loglike = np.sum(stats.gamma.logpdf(data, a=fitted_params_gamma[0], loc=fitted_params_gamma[1],
                                        scale=fitted_params_gamma[2]))
    k = 3
    return k, loglike


def log_invgauss(data, fitted_params_invgauss):
    loglike = np.sum(stats.invgauss.logpdf(data, mu=fitted_params_invgauss[0], loc=fitted_params_invgauss[1],
                                           scale=fitted_params_invgauss[2]))
    k = 3
    return k, loglike


def log_logistic(data, fitted_params_logistic):
    loglike = np.sum(
        stats.logistic.logpdf(data, loc=fitted_params_logistic[0], scale=fitted_params_logistic[1]))
    k = 2
    return k, loglike
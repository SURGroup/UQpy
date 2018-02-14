import scipy.stats as stats
from functools import partial
import numpy as np


########################################################################################################################
#        Define the probability distribution of the random parameters
########################################################################################################################


def normal_pdf(x):
    """ Normal density function used to generate samples using the Metropolis-Hastings Algorithm
     :math: `f(x) = \\frac{1}{(2*\\pi*\\sigma)^(1/2)}*exp(-\\frac{1}{2}*(\\frac{x-\\mu}{\\sigma})^2)`

    """
    return stats.norm.pdf(x, 0, 1)


def multivariate_pdf(x, dim):
    """ Multivariate normal density function used to generate samples using the Metropolis-Hastings Algorithm
    :math: `f(x_{1},...,x_{k}) = \\frac{1}{((2*\\pi)^{k}*\\Sigma)^(1/2)}*exp(-\\frac{1}{2}*(x-\\mu)^{T}*\\Sigma^{-1}*(x-\\mu))`

    """
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

    elif dist == 'normal_pdf':
        return partial(normal_pdf)

    elif dist == 'marginal_pdf':
        return partial(marginal_pdf)

########################################################################################################################
#        Transform the random parameters to the original space
########################################################################################################################


def transform_pdf(x, pdf, params):
    x_trans = np.zeros(shape=(x.shape[0], x.shape[1]))
    ###################################################################################
    # U(0, 1)  ---->  U(a, b)

    for i in range(x.shape[1]):
        if pdf[i] == 'Uniform':
            for j in range(x.shape[0]):
                     x_trans[j, i] = params[i][0] + (params[i][1]-params[i][0]) * x[j, i]

    ###################################################################################
    # U(0, 1)  ---->  N(μ, σ)

        elif pdf[i] == 'Normal':
            for j in range(x.shape[0]):
                    x_trans[j, i] = stats.norm.ppf(x[j, i], params[i][0], params[i][1])

    ####################################################################################

    # U(0, 1)  ---->  LN(μ, σ)
    # TODO: Transform U(0, 1) to LN(μ, σ)

    return x_trans

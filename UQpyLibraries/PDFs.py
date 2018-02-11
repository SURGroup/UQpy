import scipy.stats as stats
from functools import partial
import numpy as np

def normpdf(x):
    """ Normal density function used to generate samples using Metropolis-Hastings Algorithm
     :math: `f(x) = \\frac{1}{(2*\\pi*\\sigma)^(1/2)}*exp(-\\frac{1}{2}*(\\frac{x-\\mu}{\\sigma})^2)`

    """
    return stats.norm.pdf(x, 0, 1)


def mvnpdf(x, dim):
    """ Multivariate normal density function used to generate samples using Metropolis-Hastings Algorithm
    :math: `f(x_{1},...,x_{k}) = \\frac{1}{((2*\\pi)^{k}*\\Sigma)^(1/2)}*exp(-\\frac{1}{2}*(x-\\mu)^{T}*\\Sigma^{-1}*(x-\\mu))`

    """
    return stats.multivariate_normal.pdf(x, mean=np.zeros(dim), cov=np.identity(dim))


def marginal(x, mp):
    """
    Marginal target density used to generate samples using Modified Metropolis-Hastings Algorithm

    :math:`f(x) = \\frac{1}{\\sqrt{2\\pi\\sigma}}\\exp{-\\frac{1}{2}{\\frac{x-\\mu}{\\sigma}}^2}`

    """
    return stats.norm.pdf(x, mp[0], mp[1])


def srom(x):
    return stats.gamma.cdf(x, 2, loc=1, scale=3)


def pdf(dist):
    if dist == 'mvnpdf':
        return partial(mvnpdf)

    elif dist == 'normpdf':
        return partial(normpdf)

    elif dist == 'marginal':
        return partial(marginal)

    elif dist == 'srom1':
        return partial(srom)

    elif dist == 'srom2':
        return partial(srom)

    elif dist == 'srom3':
        return partial(srom)


def transform_pdf(x, pdf, params):
    x_trans = np.zeros(shape=(x.shape[0], x.shape[1]))
    for i in range(x.shape[1]):
            for j in range(x.shape[0]):
                x_trans[j, i] = params[i][0] +(params[i][1]-params[i][0])*x[j, i]

    return x_trans

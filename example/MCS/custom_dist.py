import numpy as np
import scipy.stats as stats

def pdf(x, params):
    return stats.weibull_min.pdf(x, params[0], params[1])

def cdf(x, params):
    return stats.weibull_min.cdf(x, params[0], params[1])

def icdf(x, params):
    return stats.weibull_min.ppf(x, params[0], params[1])

def rvs(nsamples, params):
    return stats.weibull_min.rvs(params[0], params[1], size=nsamples)

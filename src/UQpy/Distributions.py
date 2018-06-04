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
    for i in range(len(dist)):
        if type(dist).__name__ == 'str':
            if dist == 'Uniform':
                def f(x, params):
                    return stats.uniform.pdf(x, params[0], params[1])

                dist = f

            elif dist == 'Normal':
                def f(x, params):
                    return stats.norm.pdf(x, params[0], params[1])

                dist = f

            elif dist == 'Lognormal':
                def f(x, params):
                    return stats.lognorm.pdf(x, params[0], params[1])

                dist = f

            elif dist == 'Weibull':
                def f(x, params):
                    return stats.weibull_min.pdf(x, params[0], params[1])

                dist = f

            elif dist == 'Beta':
                def f(x, params):
                    return stats.weibull_min.pdf(x, params[0], params[1], params[2], params[3])

                dist = f

            elif dist == 'Exponential':
                def f(x, params):
                    return stats.expon.pdf(x, params[0], params[1])

                dist = f

            elif dist == 'Gamma':
                def f(x, params):
                    return stats.gamma.pdf(x, params[0], params[1], params[2])

                dist = f

            elif os.path.isfile('custom_dist.py') is True:
                import custom_dist
                method_to_call = getattr(custom_dist, dist)

                dist = partial(method_to_call)

            else:
                raise NotImplementedError('Unidentified pdf_type')

    return dist

# TODO: Add a library of pdfs here.


def cdf(dist):
    dir_ = os.getcwd()
    sys.path.insert(0, dir_)
    for i in range(len(dist)):
        if type(dist).__name__ == 'str':
            if dist == 'Uniform':
                def f(x, params):
                    return stats.uniform.cdf(x, params[0], params[1])

                dist = f

            elif dist == 'Normal':
                def f(x, params):
                    return stats.norm.cdf(x, params[0], params[1])

                dist = f

            elif dist == 'Lognormal':
                def f(x, params):
                    return stats.lognorm.cdf(x, params[0], params[1])

                dist = f

            elif dist == 'Weibull':
                def f(x, params):
                    return stats.weibull_min.cdf(x, params[0], params[1])

                dist = f

            elif dist == 'Beta':
                def f(x, params):
                    return stats.weibull_min.cdf(x, params[0], params[1], params[2], params[3])

                dist = f

            elif dist == 'Exponential':
                def f(x, params):
                    return stats.expon.cdf(x, params[0], params[1])

                dist = f

            elif dist == 'Gamma':
                def f(x, params):
                    return stats.gamma.cdf(x, params[0], params[1], params[2])

                dist = f

            elif os.path.isfile('custom_dist.py') is True:
                import custom_dist
                method_to_call = getattr(custom_dist, dist)

                dist = partial(method_to_call)

            else:
                raise NotImplementedError('Unidentified pdf_type')

    return dist


def inv_cdf(dist):
    dir_ = os.getcwd()
    sys.path.insert(0, dir_)
    for i in range(len(dist)):
        if type(dist).__name__ == 'str':
            if dist == 'Uniform':
                def f(x, params):
                    return stats.uniform.ppf(x, params[0], params[1])

                dist = f

            elif dist == 'Normal':
                def f(x, params):
                    return stats.norm.ppf(x, params[0], params[1])

                dist = f

            elif dist == 'Lognormal':
                def f(x, params):
                    return stats.lognorm.ppf(x, params[0], params[1])

                dist = f

            elif dist == 'Weibull':
                def f(x, params):
                    return stats.weibull_min.ppf(x, params[0], params[1])

                dist = f

            elif dist == 'Beta':
                def f(x, params):
                    return stats.weibull_min.ppf(x, params[0], params[1], params[2], params[3])

                dist = f

            elif dist == 'Exponential':
                def f(x, params):
                    return stats.expon.ppf(x, params[0], params[1])

                dist = f

            elif dist == 'Gamma':
                def f(x, params):
                    return stats.gamma.ppf(x, params[0], params[1], params[2])

                dist = f

            elif os.path.isfile('custom_dist.py') is True:
                import custom_dist
                method_to_call = getattr(custom_dist, dist)

                dist = partial(method_to_call)

            else:
                print(dist)
                raise NotImplementedError('Unidentified pdf_type')

    return dist

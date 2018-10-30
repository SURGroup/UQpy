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

"""This module contains functionality for all the distribution supported in UQpy."""

import scipy.stats as stats
from functools import partial
import os


# Authors: Dimitris G.Giovanis, Michael D. Shields
# Last Modified: 7/18/18 by Dimitris G. Giovanis


########################################################################################################################
#        Define the probability distribution of the random parameters
########################################################################################################################


class Distribution:

    def __init__(self, name, params=None):

        """
            Description:

            A module containing functions of a wide variaty of distributions that can be found in the package
            scipy.stats. The supported distributions are:
            [normal, uniform, binomial, beta, genextreme, chisquare, lognormal, gamma, exponential, cauchy, levy,
            logistic, laplace, maxwell, inverse gauss, pareto, rayleigh].
            For the assigned distribution, the distribution class provides the following functions:

                1. pdf: probability density function
                2. cdf: cumulative distribution function
                3. icdf (inverse cdf)
                4. rvs: generate random numbers (it doesn't need a point)
                5. log_pdf: logarithm of the pdf
                6. fit: Estimates the parameters of the distribution over arbitrary data
                7. moments: Calculate the first four moments of the distribution (mean, variance, skewness, kurtosis)

            Input:
                :param name: Name of distribution.
                :type: name: string

                :param params: Parameters of the distribution
                :type: params: ndarray or list

            Output:
                Objects possessing 7 aforementioned distribution functions.
        """

        self.name = name
        if params is not None:
            self.params = params

        if self.name.lower() == 'normal' or self.name.lower() == 'gaussian':

            def pdf(x, params):
                return stats.norm.pdf(x, loc=params[0], scale=params[1])
            self.pdf = partial(pdf)

            def rvs(params, nsamples):
                return stats.norm.rvs(loc=params[0], scale=params[1], size=nsamples)
            self.rvs = partial(rvs)

            def cdf(x, params):
                return stats.norm.cdf(x, loc=params[0], scale=params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.norm.ppf(x, loc=params[0], scale=params[1])
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                return stats.norm.logpdf(x, loc=params[0], scale=params[1])
            self.log_pdf = partial(log_pdf)

            def fit(x):
                return stats.norm.fit(x)
            self.fit = partial(fit)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                mean, var, skew, kurt = stats.norm.stats(scale=params[1],
                                                         loc=params[0],  moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt
                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'uniform':
            def pdf(x, params):
                loc = params[0]
                scale = params[1] - params[0]
                return stats.uniform.pdf(x, loc=loc, scale=scale)
            self.pdf = partial(pdf)

            def rvs(params, nsamples):
                loc = params[0]
                scale = params[1] - params[0]
                return stats.uniform.rvs(loc=loc, scale=scale, size=nsamples)
            self.rvs = partial(rvs)

            def cdf(x, params):
                loc = params[0]
                scale = params[1] - params[0]
                return stats.uniform.cdf(x, loc=loc, scale=scale)
            self.cdf = partial(cdf)

            def icdf(x, params):
                loc = params[0]
                scale = params[1] - params[0]
                return stats.uniform.ppf(x, loc=loc, scale=scale)
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                loc = params[0]
                scale = params[1] - params[0]
                return stats.uniform.logpdf(x, loc=loc, scale=scale)
            self.log_pdf = partial(log_pdf)

            def fit(x):
                return stats.uniform.fit(x)
            self.fit = partial(fit)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]

                mean, var, skew, kurt = stats.uniform.stats(scale=params[1]-params[0],
                                                            loc=params[0],  moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt
                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'binomial':

            def pdf(x, params):
                return stats.binom.pdf(x, n=params[0], p=params[1])
            self.pdf = partial(pdf)

            def rvs(params, nsamples):
                return stats.binom.rvs(n=params[0], p=params[1], size=nsamples)
            self.rvs = partial(rvs)

            def cdf(x, params):
                return stats.binom.cdf(x, n=params[0], p=params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.binom.ppf(x, n=params[0], p=params[1])
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                return stats.binom.logpdf(x, n=params[0], p=params[1])
            self.log_pdf = partial(log_pdf)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]

                mean, var, skew, kurt = stats.binom.stats(n=params[0],
                                                          p=params[0],  moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt
                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'beta':

            def pdf(x, params):
                return stats.beta.pdf(x, a=params[0], b=params[1])
            self.pdf = partial(pdf)

            def rvs(params, nsamples):
                return stats.beta.rvs(a=params[0], b=params[1], size=nsamples)
            self.rvs = partial(rvs)

            def cdf(x, params):
                return stats.beta.cdf(x, a=params[0], b=params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.beta.ppf(x, a=params[0], b=params[1])
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                return stats.beta.logpdf(x, a=params[0], b=params[1])
            self.log_pdf = partial(log_pdf)

            def fit(x):
                return stats.beta.fit(x)
            self.fit = partial(fit)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]

                mean, var, skew, kurt = stats.beta.stats(a=params[0],
                                                         b=params[0],  moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt
                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'genextreme':

            def pdf(x, params):
                return stats.genextreme.pdf(x, c=0, loc=params[0], scale=params[1])
            self.pdf = partial(pdf)

            def rvs(params, nsamples):
                return stats.genextreme.rvs(c=0, loc=params[0], scale=params[1], size=nsamples)
            self.rvs = partial(rvs)

            def cdf(x, params):
                return stats.genextreme.cdf(x, c=0, loc=params[0], scale=params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.genextreme.ppf(x, c=0, loc=params[0], scale=params[1])
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                return stats.genextreme.logpdf(x, c=0, loc=params[0], scale=params[1])
            self.log_pdf = partial(log_pdf)

            def fit(x):
                return stats.genextreme.fit(x)
            self.fit = partial(fit)

            def moments(params):
                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                mean, var, skew, kurt = stats.genextreme.stats(c=0, scale=params[1],
                                                               loc=params[0],  moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt
                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'chisquare':

            def pdf(x, params):
                return stats.chi2.pdf(x, df=params[0], loc=params[1], scale=params[2])
            self.pdf = partial(pdf)

            def rvs(params, nsamples):
                return stats.chi2.rvs(df=params[0], loc=params[1], scale=params[2], size=nsamples)
            self.rvs = partial(rvs)

            def cdf(x, params):
                return stats.chi2.cdf(x, df=params[0], loc=params[1], scale=params[2])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.chi2.ppf(x, df=params[0], loc=params[1], scale=params[2])
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                return stats.chi2.logpdf(x, df=params[0], loc=params[1], scale=params[2])
            self.log_pdf = partial(log_pdf)

            def fit(x):
                return stats.chi2.fit(x)
            self.fit = partial(fit)

            def moments(params):
                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                mean, var, skew, kurt = stats.chi2.stats(df=params[0], loc=params[1], scale=params[2], moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt
                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'lognormal':

            def pdf(x, params):
                import numpy as np
                return stats.lognorm.pdf(x, s=params[1], scale=np.exp(params[0]))
            self.pdf = partial(pdf)

            def rvs(params, nsamples):
                import numpy as np
                return stats.lognorm.rvs(s=params[1], scale=np.exp(params[0]), size=nsamples)
            self.rvs = partial(rvs)

            def cdf(x, params):
                import numpy as np
                return stats.lognorm.cdf(x, s=params[1], scale=np.exp(params[0]))
            self.cdf = partial(cdf)

            def icdf(x, params):
                import numpy as np
                return stats.lognorm.ppf(x, s=params[1], scale=np.exp(params[0]))
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                import numpy as np
                return stats.lognorm.logpdf(x, s=params[1], scale=np.exp(params[0]))
            self.log_pdf = partial(log_pdf)

            def fit(x):
                return stats.lognorm.fit(x)
            self.fit = partial(fit)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                mean, var, skew, kurt = stats.lognorm.stats(s=params[1],
                                                            scale=np.exp(params[0]),  moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt

                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'gamma':

            def pdf(x, params):
                return stats.gamma.pdf(x, a=params[0], loc=params[1],  scale=params[2])
            self.pdf = partial(pdf)

            def rvs(params, nsamples):
                return stats.gamma.rvs(a=params[0], loc=params[1],  scale=params[2], size=nsamples)
            self.rvs = partial(rvs)

            def cdf(x, params):
                return stats.gamma.cdf(x,  a=params[0], loc=params[1],  scale=params[2])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.gamma.ppf(x,  a=params[0], loc=params[1],  scale=params[2])
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                import numpy as np
                return stats.gamma.logpdf(x, a=params[0], loc=params[1],  scale=params[2])
            self.log_pdf = partial(log_pdf)

            def fit(x):
                return stats.gamma.fit(x)
            self.fit = partial(fit)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                mean, var, skew, kurt = stats.gamma.stats(a=params[0], loc=params[1],  scale=params[2], moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt
                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'exponential':
            def pdf(x, params):
                return stats.expon.pdf(x, params[0], scale=params[1])
            self.pdf = partial(pdf)

            def rvs(params, nsamples):
                return stats.expon.rvs(params[0], scale=params[1], size=nsamples)
            self.rvs = partial(rvs)

            def cdf(x, params):
                return stats.expon.cdf(x, params[0], scale=params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.expon.ppf(x, params[0], scale=params[1])
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                import numpy as np
                return stats.expon.logpdf(x, params[0], scale=params[1])
            self.log_pdf = partial(log_pdf)

            def fit(x):
                return stats.expon.fit(x)
            self.fit = partial(fit)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                mean, var, skew, kurt = stats.expon.stats(loc=params[0], scale=params[1], moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt
                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'cauchy':
            def pdf(x, params):
                return stats.cauchy.pdf(x, loc=params[0], scale=params[1])
            self.pdf = partial(pdf)

            def rvs(params, nsamples):
                return stats.cauchy.rvs(loc=params[0], scale=params[1], size=nsamples)
            self.rvs = partial(rvs)

            def cdf(x, params):
                return stats.cauchy.cdf(x, loc=params[0], scale=params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.cauchy.ppf(x, loc=params[0], scale=params[1])
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                return stats.cauchy.logpdf(x, loc=params[0], scale=params[1])
            self.log_pdf = partial(log_pdf)

            def fit(x):
                return stats.cauchy.fit(x)
            self.fit = partial(fit)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]

                mean, var, skew, kurt = stats.cauchy.stats(loc=params[0], scale=params[1],  moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt
                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'inv_gauss':
            def pdf(x, params):
                return stats.invgauss.pdf(x, mu=params[0], loc=params[1], scale=params[2])
            self.pdf = partial(pdf)

            def rvs(params, nsamples):
                return stats.invgauss.rvs(mu=params[0], loc=params[1], scale=params[2], size=nsamples)
            self.rvs = partial(rvs)

            def cdf(x, params):
                return stats.invgauss.cdf(x, mu=params[0], loc=params[1], scale=params[2])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.invgauss.ppf(x, mu=params[0], loc=params[1], scale=params[2])
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                return stats.invgauss.logpdf(x, mu=params[0], loc=params[1], scale=params[2])
            self.log_pdf = partial(log_pdf)

            def fit(x):
                return stats.invgauss.fit(x)
            self.fit = partial(fit)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                mean, var, skew, kurt = stats.invgauss.stats(mu=params[0], loc=params[1], scale=params[2],
                                                             moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt
                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'logistic':
            def pdf(x, params):
                return stats.logistic.pdf(x, loc=params[0], scale=params[1])
            self.pdf = partial(pdf)

            def rvs(params, nsamples):
                return stats.logistic.rvs(loc=params[0], scale=params[1], size=nsamples)
            self.rvs = partial(rvs)

            def cdf(x, params):
                return stats.logistic.cdf(x, loc=params[0], scale=params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.logistic.ppf(x, loc=params[0], scale=params[1])
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                return stats.logistic.logpdf(x, loc=params[0], scale=params[1])
            self.log_pdf = partial(log_pdf)

            def fit(x):
                return stats.logistic.fit(x)
            self.fit = partial(fit)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                mean, var, skew, kurt = stats.logistic.stats(loc=params[0], scale=params[1], moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt
                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'pareto':
            def pdf(x, params):
                return stats.pareto.pdf(x, b=params[0], loc=params[1], scale=params[2])
            self.pdf = partial(pdf)

            def rvs(params, nsamples):
                return stats.pareto.rvs(b=params[0], loc=params[1], scale=params[2], size=nsamples)
            self.rvs = partial(rvs)

            def cdf(x, params):
                return stats.pareto.cdf(x, b=params[0], loc=params[1], scale=params[2])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.pareto.ppf(x, b=params[0], loc=params[1], scale=params[2])
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                return stats.pareto.logpdf(x, b=params[0], loc=params[1], scale=params[2])
            self.log_pdf = partial(log_pdf)

            def fit(x):
                return stats.pareto.fit(x)
            self.fit = partial(fit)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                mean, var, skew, kurt = stats.pareto.stats(b=params[0], loc=params[0], scale=params[1], moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt
                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'rayleigh':
            def pdf(x, params):
                return stats.rayleigh.pdf(x, loc=params[0], scale=params[1])
            self.pdf = partial(pdf)

            def rvs(params, nsamples):
                return stats.rayleigh.rvs(loc=params[0], scale=params[1], size=nsamples)
            self.rvs = partial(rvs)

            def cdf(x, params):
                return stats.rayleigh.cdf(x, loc=params[0], scale=params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.rayleigh.ppf(x, loc=params[0], scale=params[1])
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                return stats.rayleigh.logpdf(x, loc=params[0], scale=params[1])
            self.log_pdf = partial(log_pdf)

            def fit(x):
                return stats.rayleigh.fit(x)
            self.fit = partial(fit)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                mean, var, skew, kurt = stats.rayleigh.stats(loc=params[0], scale=params[1], moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt
                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'levy':
            def pdf(x, params):
                return stats.levy.pdf(x, loc=params[0], scale=params[1])
            self.pdf = partial(pdf)

            def rvs(params, nsamples):
                return stats.levy.rvs(loc=params[0], scale=params[1], size=nsamples)
            self.rvs = partial(rvs)

            def cdf(x, params):
                return stats.levy.cdf(x, loc=params[0], scale=params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.levy.ppf(x, loc=params[0], scale=params[1])
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                return stats.levy.logpdf(x, loc=params[0], scale=params[1])
            self.log_pdf = partial(log_pdf)

            def fit(x):
                return stats.levy.fit(x)
            self.fit = partial(fit)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                mean, var, skew, kurt = stats.levy.stats(loc=params[0], scale=params[1], moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt
                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'laplace':
            def pdf(x, params):
                return stats.laplace.pdf(x, loc=params[0], scale=params[1])
            self.pdf = partial(pdf)

            def rvs(params, nsamples):
                return stats.laplace.rvs(loc=params[0], scale=params[1], size=nsamples)
            self.rvs = partial(rvs)

            def cdf(x, params):
                return stats.laplace.cdf(x, loc=params[0], scale=params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.laplace.ppf(x, loc=params[0], scale=params[1])
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                return stats.laplace.logpdf(x, loc=params[0], scale=params[1])
            self.log_pdf = partial(log_pdf)

            def fit(x):
                return stats.laplace.fit(x)
            self.fit = partial(fit)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                mean, var, skew, kurt = stats.laplace.stats(loc=params[0], scale=params[1], moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt
                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'maxwell':
            def pdf(x, params):
                return stats.maxwell.pdf(x, loc=params[0], scale=params[1])
            self.pdf = partial(pdf)

            def rvs(params, nsamples):
                return stats.maxwell.rvs(loc=params[0], scale=params[1], size=nsamples)
            self.rvs = partial(rvs)

            def cdf(x, params):
                return stats.maxwell.cdf(x, loc=params[0], scale=params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.maxwell.ppf(x, loc=params[0], scale=params[1])
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                return stats.maxwell.logpdf(x, loc=params[0], scale=params[1])
            self.log_pdf = partial(log_pdf)

            def fit(x):
                return stats.maxwell.fit(x)
            self.fit = partial(fit)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                mean, var, skew, kurt = stats.maxwell.stats(loc=params[0], scale=params[1], moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt
                return y

            self.moments = partial(moments)

        elif os.path.isfile('custom_dist.py') is True:
            import custom_dist
            self.name = 'custom'
            self.pdf = getattr(custom_dist, 'pdf', 'Attribute not defined.')
            self.cdf = getattr(custom_dist, 'cdf', 'Attribute not defined.')
            self.icdf = getattr(custom_dist, 'icdf', 'Attribute not defined.')
            self.log_pdf = getattr(custom_dist, 'log_pdf', 'Attribute not defined.')
            self.rvs = getattr(custom_dist, 'rvs', 'Attribute not defined.')
            self.fit = getattr(custom_dist, 'fit', 'Attribute not defined.')
            self.moments = getattr(custom_dist, 'moments', 'Attribute not defined.')


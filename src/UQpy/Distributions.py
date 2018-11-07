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
import numpy as np


# Authors: Dimitris G.Giovanis, Michael D. Shields
# Last Modified: 7/18/18 by Dimitris G. Giovanis


########################################################################################################################
#        Define the probability distribution of the random parameters
########################################################################################################################

def get_supported_distributions(print_ = False):
    supported_distributions = ['normal', 'uniform', 'binomial', 'beta', 'genextreme', 'chisquare', 'lognormal',
                               'gamma', 'inv_gamma', 'exponential', 'cauchy', 'levy', 'logistic', 'laplace', 'maxwell',
                               'inv_gauss', 'pareto', 'rayleigh']
    if print_:
        print(supported_distributions)
    return supported_distributions


class DistributionFromMarginals:

    def __init__(self, name, parameters=None):
        self.name = name
        self.params = parameters
        self.ndims = len(self.name)
        self.distributions = [Distribution(name=self.name[i],parameters=self.params[i]) for i in range(self.ndims)]

    def pdf(self, x, params):
        if np.size(x) == self.ndims:
            x = x.reshape((1, self.ndims))
        return np.prod([self.distributions[i].pdf(x[:,i], params[i]) for i in range(self.ndims)])

    def log_pdf(self, x, params):
        if np.size(x) == self.ndims:
            x = x.reshape((1, self.ndims))
        return np.sum([self.distributions[i].log_pdf(x[:,i], params[i]) for i in range(self.ndims)])

    def rvs(self, params, size=1):
        rvs_list = []
        for i in range(self.ndims):
            rvs_list.append(np.array(self.distributions[i].rvs(params[i])).reshape((-1,1)))
        return np.concatenate(rvs_list, axis=1)


class Distribution:

    def __init__(self, name, parameters=None):

        """
            Description:

            A module containing functions of a wide variaty of distributions that can be found in the package
            scipy.stats. The supported distributions are:
            [normal, uniform, binomial, beta, genextreme, chisquare, lognormal, gamma, exponential, cauchy, levy,
            logistic, laplace, maxwell, inverse gauss, pareto, rayleigh].
            For the assigned distribution, for a point you can estimate:

                1. pdf: probability density function
                2. cdf: cumulative distribution function
                3. icdf (inverse cdf)
                4. rvs: generate random numbers (it doesn't need a point)
                5. log_pdf: logarithm of the pdf
                6. fit: Estimates the parameters of the distribution over arbitrary data
                7. moments: Calculate the first four moments of the distribution (mean, variance, skewness, kyrtosis)

            Input:
                :param name: Name of distribution.
                :type: name: string

                :param parameters: Parameters of the distribution
                :type: parameters: ndarray

            Output:
                A handler pointing to the 17 aforementioned distribution functions.
        """

        self.name = name
        self.params = parameters

        if self.name.lower() == 'normal' or self.name.lower() == 'gaussian':

            self.n_params = 2

            def pdf(x, params):
                return stats.norm.pdf(x, loc=params[0], scale=params[1])
            self.pdf = partial(pdf)

            def rvs(params):
                return stats.norm.rvs(loc=params[0], scale=params[1])
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

            self.n_params = 2

            def pdf(x, params):
                loc = params[0]
                scale = params[1] - params[0]
                return stats.uniform.pdf(x, loc=loc, scale=scale)
            self.pdf = partial(pdf)

            def rvs(params):
                loc = params[0]
                scale = params[1] - params[0]
                return stats.uniform.rvs(loc=loc, scale=scale)
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

            self.n_params = 2

            def pdf(x, params):
                return stats.binom.pdf(x, n=params[0], p=params[1])
            self.pdf = partial(pdf)

            def rvs(params):
                return stats.binom.rvs(n=params[0], p=params[1])
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

            self.n_params = 2

            def pdf(x, params):
                return stats.beta.pdf(x, a=params[0], b=params[1])
            self.pdf = partial(pdf)

            def rvs(params):
                return stats.beta.rvs(a=params[0], b=params[1])
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

        elif self.name.lower() == 'gumbel_r':

            self.n_params = 2

            def pdf(x, params):
                return stats.genextreme.pdf(x, c=0, loc=params[0], scale=params[1])
            self.pdf = partial(pdf)

            def rvs(params):
                return stats.genextreme.rvs(c=0, loc=params[0], scale=params[1])
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

            self.n_params = 3

            def pdf(x, params):
                return stats.chi2.pdf(x, df=params[0], loc=params[1], scale=params[2])
            self.pdf = partial(pdf)

            def rvs(params):
                return stats.chi2.rvs(df=params[0], loc=params[1], scale=params[2])
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
            self.n_params = 3

            def pdf(x, params):
                import numpy as np
                return stats.lognorm.pdf(x, s=params[1], loc=params[2], scale=np.exp(params[0]))

            self.pdf = partial(pdf)

            def rvs(params):
                import numpy as np
                return stats.lognorm.rvs(s=params[1], loc=params[2], scale=np.exp(params[0]))
            self.rvs = partial(rvs)

            def cdf(x, params):
                import numpy as np
                return stats.lognorm.cdf(x, s=params[1], loc=params[2], scale=np.exp(params[0]))

            self.cdf = partial(cdf)

            def icdf(x, params):
                import numpy as np
                return stats.lognorm.ppf(x, s=params[1], loc=params[2], scale=np.exp(params[0]))
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                import numpy as np
                return stats.lognorm.logpdf(x, s=params[1], loc=params[2], scale=np.exp(params[0]))
            self.log_pdf = partial(log_pdf)

            def fit(x):
                import numpy as np
                params = stats.lognorm.fit(x, floc=0)
                loc = params[1]
                s = params[0]
                scale = np.log(params[2])
                return list([scale, s, loc])
            self.fit = partial(fit)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                mean, var, skew, kurt = stats.lognorm.stats(s=params[1], loc=params[2], scale=np.exp(params[0]),
                                                            moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt

                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'gamma':

            self.n_params = 3

            def pdf(x, params):
                return stats.gamma.pdf(x, a=params[0], loc=params[1],  scale=params[2])
            self.pdf = partial(pdf)

            def rvs(params):
                return stats.gamma.rvs(a=params[0], loc=params[1],  scale=params[2])
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

        elif self.name.lower() == 'inv_gamma':

            self.n_params = 3

            def pdf(x, params):
                return stats.invgamma.pdf(x, a=params[0], loc=params[1],  scale=params[2])
            self.pdf = partial(pdf)

            def rvs(params):
                return stats.invgamma.rvs(a=params[0], loc=params[1],  scale=params[2])
            self.rvs = partial(rvs)

            def cdf(x, params):
                return stats.invgamma.cdf(x,  a=params[0], loc=params[1],  scale=params[2])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.invgamma.ppf(x,  a=params[0], loc=params[1],  scale=params[2])
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                return stats.invgamma.logpdf(x, a=params[0], loc=params[1],  scale=params[2])
            self.log_pdf = partial(log_pdf)

            def fit(x):
                return stats.invgamma.fit(x)
            self.fit = partial(fit)

            def moments(params):

                y = [np.nan, np.nan, np.nan, np.nan]
                mean, var, skew, kurt = stats.invgamma.stats(a=params[0], loc=params[1],  scale=params[2], moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt
                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'exponential':

            self.n_params = 2

            def pdf(x, params):
                return stats.expon.pdf(x, loc=params[0], scale=1/params[1])
            self.pdf = partial(pdf)

            def rvs(params):
                return stats.expon.rvs(loc=params[0], scale=1/params[1])
            self.rvs = partial(rvs)

            def cdf(x, params):
                return stats.expon.cdf(x, loc=params[0], scale=1/params[1])
            self.cdf = partial(cdf)

            def icdf(x, params):
                return stats.expon.ppf(x, loc=params[0], scale=1/params[1])
            self.icdf = partial(icdf)

            def log_pdf(x, params):
                import numpy as np
                return stats.expon.logpdf(x, loc=params[0], scale=1/params[1])
            self.log_pdf = partial(log_pdf)

            def fit(x):
                return stats.expon.fit(x)
            self.fit = partial(fit)

            def moments(params):

                import numpy as np
                y = [np.nan, np.nan, np.nan, np.nan]
                mean, var, skew, kurt = stats.expon.stats(loc=params[0], scale=1/params[1], moments='mvsk')
                y[0] = mean
                y[1] = var
                y[2] = skew
                y[3] = kurt
                return y

            self.moments = partial(moments)

        elif self.name.lower() == 'cauchy':

            self.n_params = 2

            def pdf(x, params):
                return stats.cauchy.pdf(x, loc=params[0], scale=params[1])
            self.pdf = partial(pdf)

            def rvs(params):
                return stats.cauchy.rvs(loc=params[0], scale=params[1])
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

            self.n_params = 3

            def pdf(x, params):
                return stats.invgauss.pdf(x, mu=params[0], loc=params[1], scale=params[2])
            self.pdf = partial(pdf)

            def rvs(params):
                return stats.invgauss.rvs(mu=params[0], loc=params[1], scale=params[2])
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

            self.n_params = 2

            def pdf(x, params):
                return stats.logistic.pdf(x, loc=params[0], scale=params[1])
            self.pdf = partial(pdf)

            def rvs(params):
                return stats.logistic.rvs(loc=params[0], scale=params[1])
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

            self.n_params = 3

            def pdf(x, params):
                return stats.pareto.pdf(x, b=params[0], loc=params[1], scale=params[2])
            self.pdf = partial(pdf)

            def rvs(params):
                return stats.pareto.rvs(b=params[0], loc=params[1], scale=params[2])
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

            self.n_params = 2

            def pdf(x, params):
                return stats.rayleigh.pdf(x, loc=params[0], scale=params[1])
            self.pdf = partial(pdf)

            def rvs(params):
                return stats.rayleigh.rvs(loc=params[0], scale=params[1])
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

            self.n_params = 2

            def pdf(x, params):
                return stats.levy.pdf(x, loc=params[0], scale=params[1])
            self.pdf = partial(pdf)

            def rvs(params):
                return stats.levy.rvs(loc=params[0], scale=params[1])
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

            self.n_params = 2

            def pdf(x, params):
                return stats.laplace.pdf(x, loc=params[0], scale=params[1])
            self.pdf = partial(pdf)

            def rvs(params):
                return stats.laplace.rvs(loc=params[0], scale=params[1])
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

            self.n_params = 2

            def pdf(x, params):
                return stats.maxwell.pdf(x, loc=params[0], scale=params[1])
            self.pdf = partial(pdf)

            def rvs(params):
                return stats.maxwell.rvs(loc=params[0], scale=params[1])
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

        else:
            name_file = name+'py'
            if not os.path.isfile(name_file):
                raise ValueError('The distribution should either be supported or user-defined in a name.py file')
            import name_file
            self.pdf = getattr(name_file, 'pdf')
            self.cdf = getattr(name_file, 'cdf')
            self.icdf = getattr(name_file, 'icdf')
            self.log_pdf = getattr(name_file, 'log_pdf')
            self.fit = getattr(name_file, 'fit')
            self.moments = getattr(name_file, 'moments')



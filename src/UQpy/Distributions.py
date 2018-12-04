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
import os
import numpy as np


# Authors: Dimitris G.Giovanis, Michael D. Shields
# Last Modified: 12/4/18 by Dimitris G. Giovanis


########################################################################################################################
#        Define the probability distribution of the random parameters
########################################################################################################################

class Distribution:
    """
        Description:

        Main distribution class available to the user. The user can define a probability distribution by providing:
        - a name that points to a univariate/multivariate distribution (see supported distributions in SubDistribution
        class or custom distribution)
        - a list of names that points to a list of univariate distributions. In that case, a multivariate distribution
        is built for which all dimensions are independent and given by Distribution(name)
        - a list of names and a copula, in that case a multivariate distribution is built using Distribution(name)
        for the marginal pdfs, while the dependence structure is given by the copula.

        The following methods are defined:

            1. pdf: probability density function
            2. cdf: cumulative distribution function
            3. icdf inverse cumulative distribution function
            4. rvs: generate random numbers (it doesn't need a point)
            5. log_pdf: logarithm of the pdf
            6. fit: Estimates the parameters of the distribution over arbitrary data
            7. moments: Calculate the first four moments of the distribution (mean, variance, skewness, kyrtosis)

        Input:
            :param name: Name of distribution.
            :type: name: string or list of strings

            :param copula: copula to create dependence within dimensions, used only if name is a list
            :type: copula: str or None (default None)

        Output:
            A handler pointing to a distribution and its associated methods.
    """

    def __init__(self, name=None, copula=None):

        if name is None:
            raise ValueError('UQpy error: A Distribution name must be provided!')
        if not isinstance(name, str) and not (isinstance(name, list) and isinstance(name[0], str)):
            raise ValueError('UQpy error: name must be a string or a list of strings.')
        self.name = name

        if copula is not None:
            if not isinstance(copula, str):
                raise ValueError('UQpy error: when provided, copula should be a string.')
            if isinstance(name, str):
                raise ValueError('UQpy error: it does not make sense to define a copula when name is a single string.')
            self.copula = Copula(copula_name=copula, dist_name=self.name)
        else:
            self.copula = None

        # Compute n_params for the distribution, is it necessary?
        # if isinstance(self.name, str):
        #     self.n_params = SubDistribution(name=self.name).n_params()
        # elif isinstance(self.name, list):
        #     n_params_i = [SubDistribution(name=name_i).n_params() for name_i in self.name]
        #     if self.copula is None:
        #         self.n_params = sum(n_params_i)

    def pdf(self, x, params, copula_params=None):

        if isinstance(self.name, str):
            return SubDistribution(name=self.name).pdf(x, params)
        elif isinstance(self.name, list):
            if (x.shape[1] != len(self.name)) or (len(params) != len(self.name)):
                raise ValueError('UQpy error: Inconsistent dimensions')
            prod_pdf = 1
            for i in range(len(self.name)):
                prod_pdf = prod_pdf * SubDistribution(self.name[i]).pdf(x[:, i], params[i])
            if self.copula is None:
                return prod_pdf
            else:
                c = self.copula.pdf_term(x=x, dist_params=params, copula_params=copula_params)
                return prod_pdf * c

    def log_pdf(self, x, params):

        if isinstance(self.name, str):
            return SubDistribution(name=self.name).log_pdf(x, params)
        elif isinstance(self.name, list):
            if (x.shape[1] != len(self.name)) or (len(params) != len(self.name)):
                raise ValueError('UQpy error: Inconsistent dimensions')
            if self.copula is None:
                sum_log_pdf = 0
                for i in range(len(self.name)):
                    sum_log_pdf = sum_log_pdf + SubDistribution(self.name[i]).log_pdf(x[:, i], params[i])
                return sum_log_pdf

    def rvs(self, params, nsamples=1):

        if isinstance(self.name, str):
            return SubDistribution(name=self.name).rvs(params, nsamples)
        elif isinstance(self.name, list):
            if len(params) != len(self.name):
                raise ValueError('UQpy error: Inconsistent dimensions')
            if self.copula is None:
                rvs = np.zeros((nsamples, len(self.name)))
                for i in range(len(self.name)):
                    rvs[:, i] = SubDistribution(self.name[i]).rvs(params[i], nsamples)
                return rvs

    def cdf(self, x, params):

        if isinstance(self.name, str):
            return SubDistribution(name=self.name).cdf(x, params)
        elif isinstance(self.name, list):
            if (len(params) != len(self.name)) or (x.shape[1] != len(self.name)):
                raise ValueError('UQpy error: Inconsistent dimensions')
            if self.copula is None:
                icdfs = []
                for i in range(len(self.name)):
                    icdfs.append(SubDistribution(self.name[i]).icdf(x[:, i], params[i]))
                return icdfs

    def icdf(self, x, params):

        if isinstance(self.name, str):
            return SubDistribution(name=self.name).icdf(x, params)
        elif isinstance(self.name, list):
            if (len(params) != len(self.name)) or (x.shape[1] != len(self.name)):
                raise ValueError('UQpy error: Inconsistent dimensions')
            if self.copula is None:
                icdfs = []
                for i in range(len(self.name)):
                    icdfs.append(SubDistribution(self.name[i]).icdf(x[:, i], params[i]))
                return icdfs

    def fit(self, x):

        if isinstance(self.name, str):
            return SubDistribution(name=self.name).fit(x)
        elif isinstance(self.name, list):
            if x.shape[1] != len(self.name):
                raise ValueError('UQpy error: Inconsistent dimensions')
            if self.copula is None:
                params_fit = []
                for i in range(len(self.name)):
                    params_fit.append(SubDistribution(self.name[i]).fit(x[:, i]))
                return params_fit

    def moments(self, params):

        if isinstance(self.name, str):
            return SubDistribution(name=self.name).moments(params)
        elif isinstance(self.name, list):
            if len(params) != len(self.name):
                raise ValueError('UQpy error: Inconsistent dimensions')
            if self.copula is None:
                mean, var, skew, kurt = [0]*len(self.name), [0]*len(self.name), [0]*len(self.name), [0]*len(self.name),
                for i in range(len(self.name)):
                    mean[i], var[i], skew[i], kurt[i] = SubDistribution(self.name[i]).moments(params[i])
                return mean, var, skew, kurt


class Copula:

    def __init__(self, copula_name=None, dist_name=None):

        if copula_name is None:
            raise ValueError('UQpy error: a name must be provided for the copula.')
        if copula_name.lower() not in ['gumbel']:
            raise ValueError('UQpy error: supported copulas are Gumbel, ...')
        self.copula_name = copula_name
        self.dist_name = dist_name

    def pdf_term(self, x, dist_params, copula_params):

        if self.copula_name.lower() == 'gumbel':
            if x.shape[1] > 2:
                raise ValueError('Maximum dimension for the Gumbel Copula is 2.')
            if not isinstance(copula_params, list):
                copula_params = [copula_params]
            if copula_params[0] < 1:
                raise ValueError('The parameter for Gumbel copula must be defined in [1, +oo)')
            if copula_params[0] == 1:
                return np.ones(x.shape[0])
            else:
                uu = np.zeros_like(x)
                for i in range(uu.shape[1]):
                    uu[:, i] = SubDistribution(self.dist_name[i]).cdf(x[:, i], dist_params[i])

                u = uu[:, 0]
                v = uu[:, 1]
                c = np.exp(-((-np.log(u)) ** copula_params[0]+(-np.log(v)) ** copula_params[0]) **
                            (1/copula_params[0]))

                c_ = c * 1/u*1/v*((-np.log(u)) ** copula_params[0]+(-np.log(v)) ** copula_params[0]) ** \
                     (-2 + 2/copula_params[0]) * (np.log(u) * np.log(v)) ** (copula_params[0]-1) *\
                     (1 + (copula_params[0] - 1) * ((-np.log(u)) ** copula_params[0] +
                                                    (-np.log(v)) ** copula_params[0]) ** (-1/copula_params[0]))
                return c_
        else:
            raise ValueError('Copula type not supported!')


class SubDistribution:

    def __init__(self, name=None):

        """
            Description:

            A module containing functions of a wide variaty of known distributions that can be found in the package
            scipy.stats. This subclass is called by the Distribution class whenever appropriate.

            The supported univariate distributions are:
            [normal, uniform, binomial, beta, genextreme, chisquare, lognormal, gamma, exponential, cauchy, levy,
            logistic, laplace, maxwell, inverse gauss, pareto, rayleigh].

            The supported multivariate distributions are:
            [mvnormal].

            However, a user-defined distribution can be used in UQpy provided a python script .py containing the
            required functions.

            For the assigned distribution, the following methods are defined:

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

            Output:
                A handler pointing to the aforementioned distribution functions.
        """

        self.name = name

        if self.name is None:
            raise ValueError('Error: A Distribution name must be provided!')

    def pdf(self, x, params):
        if self.name.lower() == 'normal' or self.name.lower() == 'gaussian':
            return stats.norm.pdf(x, loc=params[0], scale=params[1])
        if self.name.lower() == 'standard_normal':
            return stats.norm.pdf(x, loc=0.0, scale=1.0)
        elif self.name.lower() == 'uniform':
            return stats.uniform.pdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'binomial':
            return stats.binom.pdf(x, n=params[0], p=params[1])
        elif self.name.lower() == 'beta':
            return stats.beta.pdf(x, a=params[0], b=params[1])
        elif self.name.lower() == 'gumbel_r':
            return stats.genextreme.pdf(x, c=0, loc=params[0], scale=params[1])
        elif self.name.lower() == 'chisquare':
            return stats.chi2.pdf(x, df=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'lognormal':
            return stats.lognorm.pdf(x, s=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'gamma':
            return stats.gamma.pdf(x, a=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'exponential':
            return stats.expon.pdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'cauchy':
            return stats.cauchy.pdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'inv_gauss':
            return stats.invgauss.pdf(x, mu=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'logistic':
            return stats.logistic.pdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'pareto':
            return stats.pareto.pdf(x, b=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'rayleigh':
            return stats.rayleigh.pdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'levy':
            return stats.levy.pdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'laplace':
            return stats.laplace.pdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'maxwell':
            return stats.maxwell.pdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'mvnormal':
            return stats.multivariate_normal.pdf(x, mean=params[0], cov=params[1])
        else:
            file_name = os.path.join(self.name + '.py')
            if os.path.isfile(file_name):
                import importlib
                custom_dist = importlib.import_module(self.name)
            else:
                raise FileExistsError()

            if hasattr(custom_dist, 'pdf'):
                pdf = getattr(custom_dist, 'pdf')
                return pdf(x, params)
            else:
                return getattr(custom_dist, 'pdf', 'Attribute pdf not defined.')

    def rvs(self, params, nsamples):
        if self.name.lower() == 'normal' or self.name.lower() == 'gaussian':
            return stats.norm.rvs(loc=params[0], scale=params[1], size=nsamples)
        if self.name.lower() == 'standard_normal':
            return stats.norm.rvs(loc=0.0, scale=1.0, size=nsamples)
        elif self.name.lower() == 'uniform':
            return stats.uniform.rvs(loc=params[0], scale=params[1], size=nsamples)
        elif self.name.lower() == 'binomial':
            return stats.binom.rvs(n=params[0], p=params[1], size=nsamples)
        elif self.name.lower() == 'beta':
            return stats.beta.rvs(a=params[0], b=params[1], size=nsamples)
        elif self.name.lower() == 'gumbel_r':
            return stats.genextreme.rvs(c=0, loc=params[0], scale=params[1], size=nsamples)
        elif self.name.lower() == 'chisquare':
            return stats.chi2.rvs(df=params[0], loc=params[1], scale=params[2], size=nsamples)
        elif self.name.lower() == 'lognormal':
            return stats.lognorm.rvs(s=params[0], loc=params[1], scale=params[2], size=nsamples)
        elif self.name.lower() == 'gamma':
            return stats.gamma.rvs(a=params[0], loc=params[1], scale=params[2], size=nsamples)
        elif self.name.lower() == 'exponential':
            return stats.expon.rvs(loc=params[0], scale=params[1], size=nsamples)
        elif self.name.lower() == 'cauchy':
            return stats.cauchy.rvs(loc=params[0], scale=params[1], size=nsamples)
        elif self.name.lower() == 'inv_gauss':
            return stats.invgauss.rvs(mu=params[0], loc=params[1], scale=params[2], size=nsamples)
        elif self.name.lower() == 'logistic':
            return stats.logistic.rvs(loc=params[0], scale=params[1], size=nsamples)
        elif self.name.lower() == 'pareto':
            return stats.pareto.rvs(b=params[0], loc=params[1], scale=params[2], size=nsamples)
        elif self.name.lower() == 'rayleigh':
            return stats.rayleigh.rvs(loc=params[0], scale=params[1], size=nsamples)
        elif self.name.lower() == 'levy':
            return stats.levy.rvs(loc=params[0], scale=params[1], size=nsamples)
        elif self.name.lower() == 'laplace':
            return stats.laplace.rvs(loc=params[0], scale=params[1], size=nsamples)
        elif self.name.lower() == 'maxwell':
            return stats.maxwell.rvs(loc=params[0], scale=params[1], size=nsamples)
        elif self.name.lower() == 'mvnormal':
            return stats.multivariate_normal.rvs(mean=params[0], cov=params[1], size=nsamples)
        else:
            file_name = os.path.join(self.name + '.py')
            if os.path.isfile(file_name):
                import importlib
                custom_dist = importlib.import_module(self.name)
            else:
                raise FileExistsError()

            if hasattr(custom_dist, 'rvs'):
                rvs = getattr(custom_dist, 'rvs')
                return rvs(nsamples, params)
            else:
                return getattr(custom_dist, 'rvs', 'Attribute rvs not defined.')

    def cdf(self, x, params):
        if self.name.lower() == 'normal' or self.name.lower() == 'gaussian':
            return stats.norm.cdf(x, loc=params[0], scale=params[1])
        if self.name.lower() == 'standard_normal':
            return stats.norm.cdf(x, loc=0.0, scale=1.0)
        elif self.name.lower() == 'uniform':
            return stats.uniform.cdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'binomial':
            return stats.binom.cdf(x, n=params[0], p=params[1])
        elif self.name.lower() == 'beta':
            return stats.beta.cdf(x, a=params[0], b=params[1])
        elif self.name.lower() == 'gumbel_r':
            return stats.genextreme.cdf(x, c=0, loc=params[0], scale=params[1])
        elif self.name.lower() == 'chisquare':
            return stats.chi2.cdf(x, df=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'lognormal':
            return stats.lognorm.cdf(x, s=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'gamma':
            return stats.gamma.cdf(x, a=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'exponential':
            return stats.expon.cdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'cauchy':
            return stats.cauchy.cdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'inv_gauss':
            return stats.invgauss.cdf(x, mu=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'logistic':
            return stats.logistic.cdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'pareto':
            return stats.pareto.cdf(x, b=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'rayleigh':
            return stats.rayleigh.cdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'levy':
            return stats.levy.cdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'laplace':
            return stats.laplace.cdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'maxwell':
            return stats.maxwell.cdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'maxwell':
            return stats.multivariate_normal.cdf(x, mean=params[0], cov=params[1])
        else:
            file_name = os.path.join(self.name + '.py')
            if os.path.isfile(file_name):
                import importlib
                custom_dist = importlib.import_module(self.name)
            else:
                raise FileExistsError()

            if hasattr(custom_dist, 'cdf'):
                cdf = getattr(custom_dist, 'cdf')
                return cdf(x, params)
            else:
                return getattr(custom_dist, 'cdf', 'Attribute cdf not defined.')

    def icdf(self, x, params):
        if self.name.lower() == 'normal' or self.name.lower() == 'gaussian':
            return stats.norm.ppf(x, loc=params[0], scale=params[1])
        if self.name.lower() == 'standard_normal':
            return stats.norm.ppf(x, loc=0.0, scale=1.0)
        elif self.name.lower() == 'uniform':
            return stats.uniform.ppf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'binomial':
            return stats.binom.ppf(x, n=params[0], p=params[1])
        elif self.name.lower() == 'beta':
            return stats.beta.ppf(x, a=params[0], b=params[1])
        elif self.name.lower() == 'gumbel_r':
            return stats.genextreme.ppf(x, c=0, loc=params[0], scale=params[1])
        elif self.name.lower() == 'chisquare':
            return stats.chi2.ppf(x, df=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'lognormal':
            return stats.lognorm.ppf(x, s=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'gamma':
            return stats.gamma.ppf(x, a=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'exponential':
            return stats.expon.ppf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'cauchy':
            return stats.cauchy.ppf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'inv_gauss':
            return stats.invgauss.ppf(x, mu=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'logistic':
            return stats.logistic.ppf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'pareto':
            return stats.pareto.ppf(x, b=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'rayleigh':
            return stats.rayleigh.ppf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'levy':
            return stats.levy.ppf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'laplace':
            return stats.laplace.ppf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'maxwell':
            return stats.maxwell.ppf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'mvnormal':
            raise ValueError('UQpy error: multivariate normal does not have an icdf method.')
        else:
            file_name = os.path.join(self.name + '.py')
            if os.path.isfile(file_name):
                import importlib
                custom_dist = importlib.import_module(self.name)
            else:
                raise FileExistsError()

            if hasattr(custom_dist, 'icdf'):
                icdf = getattr(custom_dist, 'icdf')
                return icdf(x, params)
            else:
                return getattr(custom_dist, 'icdf', 'Attribute icd not defined.')

    def log_pdf(self, x, params):
        if self.name.lower() == 'normal' or self.name.lower() == 'gaussian':
            return stats.norm.logpdf(x, loc=params[0], scale=params[1])
        if self.name.lower() == 'standard_normal':
            return stats.norm.logpdf(x, loc=0.0, scale=1.0)
        elif self.name.lower() == 'uniform':
            return stats.uniform.logpdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'binomial':
            return stats.binom.log_pdf(x, n=params[0], p=params[1])
        elif self.name.lower() == 'beta':
            return stats.beta.logpdf(x, a=params[0], b=params[1])
        elif self.name.lower() == 'gumbel_r':
            return stats.genextreme.logpdf(x, c=0, loc=params[0], scale=params[1])
        elif self.name.lower() == 'chisquare':
            return stats.chi2.logpdf(x, df=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'lognormal':
            return stats.lognorm.logpdf(x, s=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'gamma':
            return stats.gamma.logpdf(x, a=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'exponential':
            return stats.expon.logpdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'cauchy':
            return stats.cauchy.logpdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'inv_gauss':
            return stats.invgauss.logpdf(x, mu=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'logistic':
            return stats.logistic.logpdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'pareto':
            return stats.pareto.logpdf(x, b=params[0], loc=params[1], scale=params[2])
        elif self.name.lower() == 'rayleigh':
            return stats.rayleigh.logpdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'levy':
            return stats.levy.logpdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'laplace':
            return stats.laplace.logpdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'maxwell':
            return stats.maxwell.logpdf(x, loc=params[0], scale=params[1])
        elif self.name.lower() == 'mvnormal':
            return stats.multivariate_normal.logpdf(x, mean=params[0], cov=params[1])
        else:
            file_name = os.path.join(self.name + '.py')
            if os.path.isfile(file_name):
                import importlib
                custom_dist = importlib.import_module(self.name)
            else:
                raise FileExistsError()

            if hasattr(custom_dist, 'log_pdf'):
                log_pdf = getattr(custom_dist, 'log_pdf')
                return log_pdf(x, params)
            else:
                return getattr(custom_dist, 'log_pdf', 'Attribute  log_pdf not defined.')

    def fit(self, x):
        if self.name.lower() == 'normal' or self.name.lower() == 'gaussian' or self.name.lower() == 'standard_normal':
            return stats.norm.fit(x)
        elif self.name.lower() == 'uniform':
            return stats.uniform.fit(x)
        elif self.name.lower() == 'binomial':
            return stats.binom.fit(x)
        elif self.name.lower() == 'beta':
            return stats.beta.fit(x)
        elif self.name.lower() == 'gumbel_r':
            return stats.genextreme.fit(x)
        elif self.name.lower() == 'chisquare':
            return stats.chi2.fit(x)
        elif self.name.lower() == 'lognormal':
            return stats.lognorm.fit(x)
        elif self.name.lower() == 'gamma':
            return stats.gamma.fit(x)
        elif self.name.lower() == 'exponential':
            return stats.expon.fit(x)
        elif self.name.lower() == 'cauchy':
            return stats.cauchy.fit(x)
        elif self.name.lower() == 'inv_gauss':
            return stats.invgauss.fit(x)
        elif self.name.lower() == 'logistic':
            return stats.logistic.fit(x)
        elif self.name.lower() == 'pareto':
            return stats.pareto.fit(x)
        elif self.name.lower() == 'rayleigh':
            return stats.rayleigh.fit(x)
        elif self.name.lower() == 'levy':
            return stats.levy.fit(x)
        elif self.name.lower() == 'laplace':
            return stats.laplace.fit(x)
        elif self.name.lower() == 'maxwell':
            return stats.maxwell.fit(x)
        elif self.name.lower() == 'mvnormal':
            raise ValueError('UQpy error: fit method for multivariate normal is not defined.')
        else:
            file_name = os.path.join(self.name + '.py')
            if os.path.isfile(file_name):
                import importlib
                custom_dist = importlib.import_module(self.name)
            else:
                raise FileExistsError()

            if hasattr(custom_dist, 'fit'):
                fit = getattr(custom_dist, 'fit')
                return fit(x)
            else:
                return getattr(custom_dist, 'fit', 'Attribute fit not defined.')

    def moments(self, params):
        y = [np.nan, np.nan, np.nan, np.nan]
        if self.name.lower() == 'normal' or self.name.lower() == 'gaussian':
            mean, var, skew, kurt = stats.norm.stats(scale=params[1],
                                                     loc=params[0], moments='mvsk')
        elif self.name.lower() == 'standard_normal':
            mean, var, skew, kurt = stats.norm.stats(scale=1.0,
                                                     loc=0.0, moments='mvsk')
        elif self.name.lower() == 'uniform':
            mean, var, skew, kurt = stats.uniform.stats(scale=params[1],
                                                        loc=params[0], moments='mvsk')
        elif self.name.lower() == 'binomial':
            mean, var, skew, kurt = stats.binom.stats(n=params[0],
                                                      p=params[1], moments='mvsk')
        elif self.name.lower() == 'beta':
            mean, var, skew, kurt = stats.beta.stats(a=params[0],
                                                     b=params[1], moments='mvsk')
        elif self.name.lower() == 'gumbel_r':
            mean, var, skew, kurt = stats.genextreme.stats(c=0, scale=params[1],
                                                           loc=params[0], moments='mvsk')
        elif self.name.lower() == 'chisquare':
            mean, var, skew, kurt = stats.chi2.stats(df=params[0], loc=params[1], scale=params[2],
                                                     moments='mvsk')
        elif self.name.lower() == 'lognormal':
            mean, var, skew, kurt = stats.lognorm.stats(s=params[0], loc=params[1], scale=params[2],
                                                        moments='mvsk')
        elif self.name.lower() == 'gamma':
            mean, var, skew, kurt = stats.gamma.stats(a=params[0], loc=params[1], scale=params[2],
                                                      moments='mvsk')
        elif self.name.lower() == 'exponential':
            mean, var, skew, kurt = stats.expon.stats(loc=params[0], scale=params[1],
                                                      moments='mvsk')
        elif self.name.lower() == 'cauchy':
            mean, var, skew, kurt = stats.cauchy.stats(loc=params[0], scale=params[1], moments='mvsk')
        elif self.name.lower() == 'inv_gauss':
            mean, var, skew, kurt = stats.invgauss.stats(mu=params[0], loc=params[1], scale=params[2],
                                                         moments='mvsk')
        elif self.name.lower() == 'logistic':
            mean, var, skew, kurt = stats.logistic.stats(loc=params[0], scale=params[1], moments='mvsk')
        elif self.name.lower() == 'pareto':
            mean, var, skew, kurt = stats.pareto.stats(b=params[0], loc=params[1], scale=params[2],
                                                       moments='mvsk')
        elif self.name.lower() == 'rayleigh':
            mean, var, skew, kurt = stats.rayleigh.stats(loc=params[0], scale=params[1], moments='mvsk')
        elif self.name.lower() == 'levy':
            mean, var, skew, kurt = stats.levy.stats(loc=params[0], scale=params[1], moments='mvsk')
        elif self.name.lower() == 'laplace':
            mean, var, skew, kurt = stats.laplace.stats(loc=params[0], scale=params[1], moments='mvsk')
        elif self.name.lower() == 'maxwell':
            mean, var, skew, kurt = stats.maxwell.stats(loc=params[0], scale=params[1], moments='mvsk')
        elif self.name.lower() == 'mvnormal':
            raise ValueError('UQpy error: moments method for multivariate normal is not defined.')
        else:
            file_name = os.path.join(self.name + '.py')
            if os.path.isfile(file_name):
                import importlib
                custom_dist = importlib.import_module(self.name)
            else:
                raise FileExistsError()
            if hasattr(custom_dist, 'moments'):
                moments = getattr(custom_dist, 'moments')
                return moments(params)
            else:
                return getattr(custom_dist, 'moments', 'Attribute moments not defined.')

        y[0] = mean
        y[1] = var
        y[2] = skew
        y[3] = kurt
        return np.array(y)

    def n_params(self):
        if self.name.lower() == 'normal' or self.name.lower() == 'gaussian' or self.name.lower() == 'standard_normal':
            return 2
        elif self.name.lower() == 'uniform':
            return 2
        elif self.name.lower() == 'binomial':
            return 2
        elif self.name.lower() == 'beta':
            return 2
        elif self.name.lower() == 'gumbel_r':
            return 2
        elif self.name.lower() == 'chisquare':
            return 3
        elif self.name.lower() == 'lognormal':
            return 3
        elif self.name.lower() == 'gamma':
            return 3
        elif self.name.lower() == 'exponential':
            return 2
        elif self.name.lower() == 'cauchy':
            return 2
        elif self.name.lower() == 'inv_gauss':
            return 3
        elif self.name.lower() == 'logistic':
            return 2
        elif self.name.lower() == 'pareto':
            return 3
        elif self.name.lower() == 'rayleigh':
            return 2
        elif self.name.lower() == 'levy':
            return 2
        elif self.name.lower() == 'laplace':
            return 2
        elif self.name.lower() == 'maxwell':
            return 2
        elif self.name.lower() == 'mvnormal':
            # raise ValueError('UQpy error: n_params for multivariate normal is not defined.')
            return None
        else:
            file_name = os.path.join(self.name + '.py')
            if os.path.isfile(file_name):
                import importlib
                custom_dist = importlib.import_module(self.name)
            else:
                raise FileExistsError()

            if hasattr(custom_dist, 'n_params'):
                n_params = getattr(custom_dist, 'n_params')
                return n_params()
            else:
                return getattr(custom_dist, 'n_params', 'Attribute n_params not defined.')

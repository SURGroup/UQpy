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

"""
This module contains functionality for all the distribution supported in UQpy.

The module currently contains the following classes:

- DistributionContinuous1D: Defines a 1-dimensional continuous probability distribution in UQpy.
- DistributionContinuousND: Defines a multivariate continuous probability distribution in UQpy.
- DistributionDiscrete1D: Defines a 1-dimensional discrete probability distribution in UQpy.
- DistributionDiscreteND: Defines a multivariate discrete probability distribution in UQpy.
- Copula: Defines a copula for modeling dependence in multivariate distributions.

"""

import scipy.stats as stats
import os
import numpy as np
from .Utilities import check_input_dims
import importlib
from types import MethodType

########################################################################################################################
#        Define the probability distribution of the random parameters
########################################################################################################################


class DistributionContinuous1D:
    """
    Define a 1-dimensional continuous probability distribution and its associated methods:
    - pdf
    - log_pdf
    - cdf
    - icdf
    - rvs
    - fit
    - moments

    **Authors:**

    Dimitris Giovanis, Audrey Olivier, Michael D. Shields

    Last Modified: 4/23/20 by Audrey Olivier
    """
    def __init__(self):
        pass

    def __pdf(self, x):
        """
        Evaluate the probability density function of a distribution at input points x.

        **Input:**

        :param x: Point(s) at which to evaluate the pdf.
        :type x: ndarray of shape (npoints,) or (npoints, 1)

        **Output/Returns:**

        :param pdf_values: Value(s) of the pdf at point(s) x.
        :type pdf_values: ndarray of shape (npoints, )
        """
        pass

    def __log_pdf(self, x):
        """
        Evaluate the logarithm of the probability density function of a distribution at input points x.

        **Input:**

        :param x: Point(s) at which to evaluate the log-pdf.
        :type x: ndarray of shape (npoints,) or (npoints, 1)

        **Output/Returns:**

        :param logpdf_values: Value(s) of the log-pdf at point(s) x.
        :type logpdf_values: ndarray of shape (npoints, )
        """
        pass

    def __cdf(self, x):
        """
        Evaluate the cumulative probability function of a distribution at input points x.

        **Input:**

        :param x: Point(s) at which to evaluate the cdf.
        :type x: ndarray of shape (npoints,) or (npoints, 1)

        **Output/Returns:**

        :param cdf_values: Value(s) of the cdf at point(s) x.
        :type cdf_values: ndarray of shape (npoints, )
        """
        pass

    def __icdf(self, x):
        """
        Evaluate the inverse of the cumulative probability function of a distribution at input points x.

        **Input:**

        :param x: Point(s) at which to evaluate the cdf.
        :type x: ndarray of shape (npoints,) or (npoints, 1)

        **Output/Returns:**

        :param icdf_values: Value(s) of the icdf at point(s) x.
        :type icdf_values: ndarray of shape (npoints, )
        """
        pass

    def __rvs(self, nsamples=1):
        """
        Sample iid realizations from the distribution.

        **Input:**

        :param nsamples: An integer providing the desired number of iid samples to be drawn.

                         Default: 1
        :type nsamples:  int

        **Output/Returns:**

        :param rv_s: Realizations from the distribution
        :type rv_s: ndarray of shape (nsamples, 1)
        """
        pass

    def __fit(self, x):
        """
        Compute the MLE parameters of a distribution from data.

        **Input:**

        :param x: Vector of data x, contains iid samples from the distribution
        :type x: ndarray of shape (npoints,) or (npoints, 1)

        **Output/Returns:**

        :param params_fit: MLE parameters.
        :type params_fit: ndarray
        """
        pass

    def __moments(self):
        """
        Compute moments (mean, variance, skewness, kurtosis).

        **Input:**

        None

        **Output/Returns:**

        :param mean: mean value
        :type mean: float

        :param var: variance
        :type var: float

        :param skew: skewness value
        :type skew: float

        :param kurt: kurtosis value
        :type kurt: float
        """
        pass

    @staticmethod
    def check_x_dimension(x):
        """
        Check the dimension of input x - must be an ndarray of shape (npoints,) or (npoints, 1)
        """
        x = np.atleast_1d(x)
        if len(x.shape) > 2 or (len(x.shape) == 2 and x.shape[1] != 1):
            raise ValueError('Wrong dimension in x.')
        return x.reshape((-1,))


########################################################################################################################
#        Univariate Continuous Distributions
########################################################################################################################


class ScipyContinuous(DistributionContinuous1D):

    def __init__(self, params):
        super().__init__()
        self.params = params

    def pdf(self, x):
        x = self.check_x_dimension(x)
        return self.get_scipy_class().pdf(x=x, **self.get_scipy_params(self.params))

    def log_pdf(self, x):
        x = self.check_x_dimension(x)
        return self.get_scipy_class().logpdf(x=x, **self.get_scipy_params(self.params))

    def cdf(self, x):
        x = self.check_x_dimension(x)
        return self.get_scipy_class().cdf(x=x, **self.get_scipy_params(self.params))

    def icdf(self, x):
        x = self.check_x_dimension(x)
        return self.get_scipy_class().ppf(x=x, **self.get_scipy_params(self.params))

    def rvs(self, nsamples):
        if not isinstance(nsamples, int) and nsamples >= 1:
            raise ValueError('Input nsamples must be an integer strictly greater than 0.')
        tmp_rvs = self.get_scipy_class().rvs(size=nsamples, **self.get_scipy_params(self.params))
        return tmp_rvs.reshape((-1, 1))

    def fit(self, x):
        x = self.check_x_dimension(x)
        return stats.lognorm.fit(data=x)

    def moments(self):
        return stats.lognorm.stats(moments='mvsk', **self.get_scipy_params(self.params))

    @staticmethod
    def get_scipy_class():
        return stats.rv_continuous

    @staticmethod
    def get_scipy_params(params):
        return {'loc': params[0], 'scale': params[1]}


class Normal(ScipyContinuous):
    """
    Normal distribution, params are [loc, scale]
    """
    def __init__(self, params):
        if len(params) != 2:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has 2 parameters: loc, scale.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.norm


class Uniform(ScipyContinuous):
    """
    Uniform distribution, params are [loc, scale]
    """
    def __init__(self, params):
        if len(params) != 2:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has 2 parameters: loc, scale.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.uniform


class Beta(ScipyContinuous):
    """
    Beta distribution, params are [a, b]
    """
    def __init__(self, params):
        if len(params) != 2:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has 2 parameters: a, b.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.beta

    @staticmethod
    def get_scipy_params(params):
        return {'a': params[0], 'b': params[1]}


class Genextreme(ScipyContinuous):
    """
    Genextreme distribution, params are [c, loc, scale]
    """
    def __init__(self, params):
        if len(params) != 3:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has 3 parameters: c, loc, scale.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.norm

    @staticmethod
    def get_scipy_params(params):
        return {'c': params[0], 'loc': params[0], 'scale': params[1]}


class Chisquare(ScipyContinuous):
    """
    Chisquare distribution, params are [df, loc, scale]
    """
    def __init__(self, params):
        if len(params) != 3:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has 3 parameters: df, loc, scale.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.chi2

    @staticmethod
    def get_scipy_params(params):
        return {'df': params[0], 'loc': params[1], 'scale': params[2]}


class Lognormal(ScipyContinuous):
    """
    Lognormal distribution, params are [s, loc, scale]
    """
    def __init__(self, params):
        if len(params) != 3:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has three parameters: s, loc, scale.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.lognorm

    @staticmethod
    def get_scipy_params(params):
        return {'s': params[0], 'loc': params[1], 'scale': params[2]}


class Gamma(ScipyContinuous):
    """
    Gamma distribution, params are [a, loc, scale]
    """
    def __init__(self, params):

        if len(params) != 3:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has 3 parameters: a, loc, scale.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.gamma

    @staticmethod
    def get_scipy_params(params):
        return {'a': params[0], 'loc': params[1], 'scale': params[2]}


class Exponential(ScipyContinuous):
    """
    Exponential distribution, params are [loc, scale]
    """
    def __init__(self, params):
        if len(params) != 2:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has 2 parameters: loc, scale.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.expon


class Cauchy(ScipyContinuous):
    """
    Cauchy distribution, params are [loc, scale]
    """
    def __init__(self, params):

        if len(params) != 2:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has 2 parameters: loc, scale.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.cauchy


class InvGauss(ScipyContinuous):
    """
    Inverse Gauss distribution, params are [mu, loc, scale]
    """
    def __init__(self, params):
        if len(params) != 3:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has 3 parameters: mu, loc, scale.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.invgauss

    @staticmethod
    def get_scipy_params(params):
        return {'mu': params[0], 'loc': params[1], 'scale': params[2]}


class Logistic(ScipyContinuous):
    """
    Logistic distribution, params are [loc, scale]
    """
    def __init__(self, params):
        if len(params) != 2:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has 2 parameters: loc, scale.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.logistic


class Pareto(ScipyContinuous):
    """
    Pareto distribution, params are [b, loc, scale]
    """
    def __init__(self, params):
        if len(params) != 3:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has 3 parameters: b, loc, scale.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.pareto

    @staticmethod
    def get_scipy_params(params):
        return {'b': params[0], 'loc': params[1], 'scale': params[2]}


class Rayleigh(ScipyContinuous):
    """
    Logistic distribution, params are [loc, scale]
    """
    def __init__(self, params):
        if len(params) != 2:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has 2 parameters: loc, scale.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.rayleigh


class Levy(ScipyContinuous):
    """
    Levy distribution, params are [loc, scale]
    """
    def __init__(self, params):
        if len(params) != 2:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has 2 parameters: loc, scale.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.levy


class Laplace(ScipyContinuous):
    """
    Laplace distribution, params are [loc, scale]
    """
    def __init__(self, params):
        if len(params) != 2:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has 2 parameters: loc, scale.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.laplace


class Maxwell(ScipyContinuous):
    """
    Maxwell distribution, params are [loc, scale]
    """
    def __init__(self, params):
        if len(params) != 2:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has 2 parameters: loc, scale.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.maxwell


class TruncNorm(ScipyContinuous):
    """
    Truncated normal distribution, params are [loc, scale]
    """
    def __init__(self, params):
        if len(params) != 4:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has 4 parameters: a, b, loc, scale.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.truncnorm

    @staticmethod
    def get_scipy_params(params):
        return {'a': params[0], 'b': params[1], 'loc': params[2], 'scale': params[3]}


########################################################################################################################
#        Multivariate Continuous Distributions
########################################################################################################################

class DistributionContinuousND:
    """
    Define a multivariate probability distribution and its associated methods.

    A multivariate distribution can be defined in various ways:
    - via direct sub-classing, see for instance the multivariate normal,
    - via a list of independent marginals, each of them being of class DistributionContinuous1D,
    - via a list of marginals and a copula (of class Copula) to account for dependency between dimensions.

    **Authors:**

    Dimitris Giovanis, Audrey Olivier, Michael D. Shields

    Last Modified: 4/23/20 by Audrey Olivier
    """
    def __init__(self):
        pass

    def __pdf(self, x):
        """
        Evaluate the probability density function of a distribution at input points x.

        **Input:**

        :param x: Point(s) at which to evaluate the pdf.
        :type x: ndarray of shape (npoints,) or (npoints, dimension)

        **Output/Returns:**

        :param pdf_values: Value(s) of the pdf at point(s) x.
        :type pdf_values: ndarray of shape (npoints, )
        """
        pass

    def __log_pdf(self, x):
        """
        Evaluate the logarithm of the probability density function of a distribution at input points x.

        **Input:**

        :param x: Point(s) at which to evaluate the log-pdf.
        :type x: ndarray of shape (npoints,) or (npoints, dimension)

        **Output/Returns:**

        :param logpdf_values: Value(s) of the log-pdf at point(s) x.
        :type logpdf_values: ndarray of shape (npoints, )
        """
        pass

    def __cdf(self, x):
        """
        Evaluate the cumulative probability function of a distribution at input points x.

        **Input:**

        :param x: Point(s) at which to evaluate the cdf.
        :type x: ndarray of shape (npoints, dimension)

        **Output/Returns:**

        :param cdf_values: Value(s) of the cdf at point(s) x.
        :type cdf_values: ndarray of shape (npoints, )
        """
        pass

    def __rvs(self, nsamples=1):
        """
        Sample iid realizations from the distribution.

        **Input:**

        :param nsamples: An integer providing the desired number of iid samples to be drawn.

                         Default: 1
        :type nsamples:  int

        **Output/Returns:**

        :param rv_s: Realizations from the distribution
        :type rv_s: ndarray of shape (nsamples, dimension)
        """
        pass

    def __fit(self, x):
        """
        Compute the MLE parameters of a distribution from data.

        **Input:**

        :param x: Vector of data x, contains iid samples from the distribution
        :type x: ndarray of shape (npoints, dimension)

        **Output/Returns:**

        :param params_fit: MLE parameters.
        :type params_fit: ndarray
        """
        pass

    def __moments(self):
        """
        Compute marginal moments (mean, variance, skewness, kurtosis).

        **Input:**

        None

        **Output/Returns:**

        :param mean: mean values
        :type mean: list

        :param var: variances
        :type var: list

        :param skew: skewness values
        :type skew: list

        :param kurt: kurtosis values
        :type kurt: list
        """
        pass

    @staticmethod
    def check_x_dimension(x, d=None):
        """
        Check the dimension of input x - must be an ndarray of shape (npoints, d)
        """
        x = np.array(x)
        if len(x.shape) != 2:
            raise ValueError('Wrong dimension in x.')
        if (d is not None) and (x.shape[1] != d):
            raise ValueError('Wrong dimension in x.')
        return x


class MVNormal(DistributionContinuousND):

    def __init__(self, params):
        if len(params) != 2:
            raise ValueError('The multivariate normal has two parameters, mean and covariance.')
        self.params = params
        super().__init__()

    def pdf(self, x):
        pdf_val = stats.multivariate_normal.pdf(x=x, mean=self.params[0], cov=self.params[1])
        return np.atleast_1d(pdf_val)

    def log_pdf(self, x):
        logpdf_val = stats.multivariate_normal.logpdf(x=x, mean=self.params[0], cov=self.params[1])
        return np.atleast_1d(logpdf_val)

    def rvs(self, nsamples):
        if not (isinstance(nsamples, int) and nsamples >= 1):
            raise ValueError('Input nsamples must be an integer > 0.')
        return stats.multivariate_normal.rvs(
            size=nsamples, mean=self.params[0], cov=self.params[1]).reshape((nsamples, -1))

    def fit(self, x):
        mle_mu = np.mean(x, axis=0)
        mle_cov = np.cov(x, rowvar=False, bias=True)
        return [mle_mu, mle_cov]


class JointInd(DistributionContinuousND):
    """
    Define a joint distribution from its independent marginals.

    Such a multivariate distribution possesses the following methods, on condition that all its univariate marginals
    also possess them: pdf, log_pdf, cdf, rvs, fit, moments.

    **Inputs:**

    :param marginals: marginal distributions
    :type marginals: list of DistributionContinuous1D objects
    """
    def __init__(self, marginals):
        super().__init__()

        # Check and save the marginals
        if not (isinstance(marginals, list) and all(isinstance(d, DistributionContinuous1D) for d in marginals)):
            raise ValueError('Input marginals must be a list of Distribution1d objects.')
        self.marginals = marginals

        # If all marginals have a method, the joint has it to
        if all(hasattr(m, 'pdf') for m in self.marginals):
            def joint_pdf(dist, x):
                x = dist.check_x_dimension(x)
                # Compute pdf of independent marginals
                pdf_val = np.prod(np.array([m.pdf(x[:, i]) for i, m in enumerate(dist.marginals)]), axis=0)
                return pdf_val
            self.pdf = MethodType(joint_pdf, self)

        if all(hasattr(m, 'log_pdf') for m in self.marginals):
            def joint_log_pdf(dist, x):
                x = dist.check_x_dimension(x)
                # Compute log-pdf of independent marginals
                logpdf_val = np.sum(np.array([m.log_pdf(x[:, i]) for i, m in enumerate(dist.marginals)]), axis=0)
                return logpdf_val
            self.log_pdf = MethodType(joint_log_pdf, self)

        if all(hasattr(m, 'cdf') for m in self.marginals):
            def joint_cdf(dist, x):
                x = dist.check_x_dimension(x)
                # Compute cdf of independent marginals
                cdf_val = np.prod(np.array([m.cdf(x[:, i]) for i, m in enumerate(dist.marginals)]), axis=0)
                return cdf_val
            self.cdf = MethodType(joint_cdf, self)

        if all(hasattr(m, 'rvs') for m in self.marginals):
            def joint_rvs(dist, nsamples=1):
                # Go through all marginals
                rv_s = np.zeros((nsamples, len(dist.marginals)))
                for i, m in enumerate(dist.marginals):
                    rv_s[:, i] = m.rvs(nsamples=nsamples).reshape((-1,))
                return rv_s
            self.rvs = MethodType(joint_rvs, self)

        if all(hasattr(m, 'fit') for m in self.marginals):
            def joint_fit(dist, x):
                x = dist.check_x_dimension(x)
                # Go through all marginals
                params = []
                for i, m in enumerate(dist.marginals):
                    params.append(m.fit(x=x[:, i]))
                return params
            self.fit = MethodType(joint_fit, self)

        if all(hasattr(m, 'moments') for m in self.marginals):
            def joint_moments(dist):
                # Go through all marginals
                mean, var, skew, kurt = [], [], [], []
                for i, m in enumerate(dist.marginals):
                    moments_i = m.moments()
                    mean.append(moments_i[0])
                    var.append(moments_i[1])
                    skew.append(moments_i[2])
                    kurt.append(moments_i[3])
                return mean, var, skew, kurt
            self.moments = MethodType(joint_moments, self)


class JointCopula(DistributionContinuousND):
    """
    Define a joint distribution from a list of marginals, potentially with a copula to introduce dependency.

    Such a multivariate distribution possesses a cdf method, and potentially a pdf and log_pdf method if the copula
    allows for it.

    **Inputs:**

    :param marginals: marginal distributions
    :type marginals: list of DistributionContinuous1D objects

    :param copula: copula
    :type copula: object of class Copula
    """
    def __init__(self, marginals, copula):
        super().__init__()

        # Check and save the marginals
        if not (isinstance(marginals, list) and all(isinstance(d, DistributionContinuous1D) for d in marginals)):
            raise ValueError('Input marginals must be a list of Distribution objects.')
        self.marginals = marginals

        # Check the copula. Also, all the marginals should have a cdf method
        if not isinstance(copula, Copula):
            raise ValueError('The input copula should be a Copula object.')
        if not all(hasattr(m, 'cdf') for m in self.marginals):
            raise ValueError('All the marginals should have a cdf method in order to define a joint with copula.')
        self.copula = copula

        # Check if methods should exist, if yes define them bound them to the object
        if hasattr(self.copula, 'evaluate_cdf'):
            def joint_cdf(dist, x):
                x = dist.check_x_dimension(x)
                # Compute cdf of independent marginals
                unif = np.array([m.cdf(x[:, i]) for i, m in enumerate(dist.marginals)]).T
                # Compute copula
                cdf_val = dist.copula.evaluate_cdf(unif=unif)
                return cdf_val
            self.cdf = MethodType(joint_cdf, self)

        if all(hasattr(m, 'pdf') for m in self.marginals) and hasattr(self.copula, 'evaluate_pdf'):
            def joint_pdf(dist, x):
                x = dist.check_x_dimension(x)
                # Compute pdf of independent marginals
                pdf_val = np.prod(np.array([m.pdf(x[:, i]) for i, m in enumerate(dist.marginals)]), axis=0)
                # Add copula term
                unif = np.array([m.cdf(x[:, i]) for i, m in enumerate(dist.marginals)]).T
                c_ = dist.copula.evaluate_pdf(unif=unif)
                return c_ * pdf_val
            self.pdf = MethodType(joint_pdf, self)

        if all(hasattr(m, 'log_pdf') for m in self.marginals) and hasattr(self.copula, 'evaluate_pdf'):
            def joint_log_pdf(dist, x):
                x = dist.check_x_dimension(x)
                # Compute pdf of independent marginals
                logpdf_val = np.sum(np.array([m.log_pdf(x[:, i]) for i, m in enumerate(dist.marginals)]), axis=0)
                # Add copula term
                unif = np.array([m.cdf(x[:, i]) for i, m in enumerate(dist.marginals)]).T
                c_ = dist.copula.evaluate_pdf(unif=unif)
                return np.log(c_) + logpdf_val
            self.log_pdf = MethodType(joint_log_pdf, self)


########################################################################################################################
#        Copulas
########################################################################################################################

class Copula:
    """
    Define a copula for a multivariate distribution whose dependence structure is defined with a copula.

    This class is used in support of the main Distribution class. The following copula are supported: Gumbel.

    **Input:**

    :param copula_params: parameters of the copula
    :type copula_params: list
    """

    def __init__(self, copula_params=None):
        self.copula_params = copula_params

    def evaluate_cdf(self, unif):
        """
        Compute the copula cdf C(u1, u2, ..., ud) for a d-variate uniform distribution.

        For a generic multivariate distribution with marginals F1, ...Fd, the joint cdf is computed as
        F(x1, ..., xd) = C(F1(x1), ..., Fd(xd)), thus one must first evaluate the marginals cdf, then evaluate the
        copula cdf.

        **Input:**

        :param unif: Points (uniformly distributed) at which to evaluate the copula cdf.
        :type unif: ndarray of shape (npoints, dimension)

        **Output/Returns**

        :param cdf_val: Copula cdf
        :type cdf_val: ndarray of shape (npoints, )
        """
        return np.prod(unif, axis=1)

    def evaluate_pdf(self, unif):
        """
        Compute the copula pdf term C_ for a d-variate uniform distribution.

        For a generic multivariate distribution with marginals pdfs f1, ..., fd, the joint pdf is computed as
        f(x1, ..., xd) = C_ * f1(x1) * ... * fd(xd). Thus one must first evaluate the marginals cdf and copula term C_,
        them multiply it by the marginal pdfs.

        **Input:**

        :param unif: Points (uniformly distributed) at which to evaluate the copula pdf term C_.
        :type unif: ndarray of shape (npoints, dimension)

        **Output/Returns**

        :param cdf_val: Copula pdf
        :type cdf_val: ndarray of shape (npoints, )
        """
        return np.ones(unif.shape[0])


class Gumbel(Copula):
    """
    Gumbel copula
    """
    def __init__(self, copula_params):
        # Check the input copula_params
        if isinstance(copula_params, (float, int)):
            copula_params = [copula_params]
        if not (isinstance(copula_params, (list, tuple)) and len(copula_params) == 1):
            raise ValueError('Input copula_params for Gumbel must be a float or list of length 1.')
        if copula_params[0] < 1:
            raise ValueError('The parameter for Gumbel copula must be defined in [1, +oo)')
        super().__init__(copula_params=copula_params)

    def evaluate_cdf(self, unif):
        if unif.shape[1] > 2:
            raise ValueError('Maximum dimension for the Gumbel Copula is 2.')
        if self.copula_params[0] == 1:
            return np.prod(unif, axis=1)

        u = unif[:, 0]
        v = unif[:, 1]
        theta = self.copula_params[0]
        cdf_val = np.exp(-((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta))

        return cdf_val

    def evaluate_pdf(self, unif):
        if unif.shape[1] > 2:
            raise ValueError('Maximum dimension for the Gumbel Copula is 2.')
        if self.copula_params[0] == 1:
            return np.ones(unif.shape[0])

        u = unif[:, 0]
        v = unif[:, 1]
        theta = self.copula_params[0]
        c = np.exp(-((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta))

        pdf_val = c * 1 / u * 1 / v * ((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (-2 + 2 / theta) \
             * (np.log(u) * np.log(v)) ** (theta - 1) * \
             (1 + (theta - 1) * ((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (-1 / theta))
        return pdf_val


class Clayton(Copula):
    """
    Clayton copula
    """
    def __init__(self, copula_params):
        # Check the input copula_params
        if isinstance(copula_params, (float, int)):
            copula_params = [copula_params]
        if not (isinstance(copula_params, (list, tuple)) and len(copula_params) == 1):
            raise ValueError('Input copula_params for Clayton must be a float or list of length 1.')
        if copula_params[0] < -1 or copula_params[0] == 0.:
            raise ValueError('The parameter for Clayton copula must be defined in [-1, +oo)\{0}')
        super().__init__(copula_params=copula_params)

    def evaluate_cdf(self, unif):
        if unif.shape[1] > 2:
            raise ValueError('Maximum dimension for the Clayton Copula is 2.')
        if self.copula_params[0] == 1:
            return np.prod(unif, axis=1)

        u = unif[:, 0]
        v = unif[:, 1]
        theta = self.copula_params[0]
        cdf_val = (np.maximum(u ** (-theta) + v ** (-theta) - 1., 0.)) ** (-1. / theta)

        return cdf_val


class Frank(Copula):
    """
    Frank copula
    """
    def __init__(self, copula_params):
        # Check the input copula_params
        if isinstance(copula_params, (float, int)):
            copula_params = [copula_params]
        if not (isinstance(copula_params, (list, tuple)) and len(copula_params) == 1):
            raise ValueError('Input copula_params for Frank must be a float or list of length 1.')
        if copula_params[0] == 0.:
            raise ValueError('The parameter for Frank copula must be defined in R\{0}')
        super().__init__(copula_params=copula_params)

    def evaluate_cdf(self, unif):
        if unif.shape[1] > 2:
            raise ValueError('Maximum dimension for the Clayton Copula is 2.')
        if self.copula_params[0] == 1:
            return np.prod(unif, axis=1)

        u = unif[:, 0]
        v = unif[:, 1]
        theta = self.copula_params[0]
        tmp_ratio = (np.exp(-theta * u) - 1.) * (np.exp(-theta * v) - 1.) / (np.exp(-theta) - 1.)
        cdf_val = -1. / theta * np.log(1. + tmp_ratio)

        return cdf_val


########################################################################################################################
#        Univariate Discrete Distributions
########################################################################################################################

class DistributionDiscrete1D:
    """
    Define a 1-dimensional probability distribution and its associated methods:
    - pmf
    - log_pmf
    - cdf
    - icdf
    - rvs
    - moments

    **Authors:**

    Dimitris Giovanis, Audrey Olivier, Michael D. Shields

    Last Modified: 4/23/20 by Audrey Olivier
    """
    def __init__(self):
        pass

    def __pmf(self, x):
        """
        Evaluate the probability mass function of a distribution at input points x.

        **Input:**

        :param x: Point(s) at which to evaluate the pmf.
        :type x: ndarray of shape (npoints,) or (npoints, 1)

        **Output/Returns:**

        :param pdf_values: Value(s) of the pmf at point(s) x.
        :type pdf_values: ndarray of shape (npoints, )
        """
        pass

    def __log_pmf(self, x):
        """
        Evaluate the logarithm of the probability mass function of a distribution at input points x.

        **Input:**

        :param x: Point(s) at which to evaluate the log-pmf.
        :type x: ndarray of shape (npoints,) or (npoints, 1)

        **Output/Returns:**

        :param logpdf_values: Value(s) of the log-pdf at point(s) x.
        :type logpdf_values: ndarray of shape (npoints, )
        """
        pass

    def __cdf(self, x):
        """
        Evaluate the cumulative probability function of a distribution at input points x.

        **Input:**

        :param x: Point(s) at which to evaluate the cdf.
        :type x: ndarray of shape (npoints,) or (npoints, 1)

        **Output/Returns:**

        :param cdf_values: Value(s) of the cdf at point(s) x.
        :type cdf_values: ndarray of shape (npoints, )
        """
        pass

    def __icdf(self, x):
        """
        Evaluate the inverse of the cumulative probability function of a distribution at input points x.

        **Input:**

        :param x: Point(s) at which to evaluate the cdf.
        :type x: ndarray of shape (npoints,) or (npoints, 1)

        **Output/Returns:**

        :param icdf_values: Value(s) of the icdf at point(s) x.
        :type icdf_values: ndarray of shape (npoints, )
        """
        pass

    def __rvs(self, nsamples=1):
        """
        Sample iid realizations from the distribution.

        **Input:**

        :param nsamples: An integer providing the desired number of iid samples to be drawn.

                         Default: 1
        :type nsamples:  int

        **Output/Returns:**

        :return rv_s: Realizations from the distribution
        :rtype rv_s: ndarray of shape (nsamples, 1)
        """
        pass

    def __moments(self):
        """
        Compute moments (mean, variance, skewness, kurtosis).

        **Input:**

        None

        **Output/Returns:**

        :param mean: mean value
        :type mean: float

        :param var: variance
        :type var: float

        :param skew: skewness value
        :type skew: float

        :param kurt: kurtosis value
        :type kurt: float
        """
        pass

    @staticmethod
    def check_x_dimension(x):
        """
        Check the dimension of input x - must be an ndarray of shape (npoints,) or (npoints, 1)
        """
        x = np.atleast_1d(x)
        if len(x.shape) > 2 or (len(x.shape) == 2 and x.shape[1] != 1):
            raise ValueError('Wrong dimension in x.')
        return x.reshape((-1,))


class ScipyDiscrete(DistributionDiscrete1D):

    def __init__(self, params):
        super().__init__()
        self.params = params

    def pmf(self, x):
        x = self.check_x_dimension(x)
        return self.get_scipy_class().pmf(x=x, **self.get_scipy_params(self.params))

    def log_pmf(self, x):
        x = self.check_x_dimension(x)
        return self.get_scipy_class().logpmf(x=x, **self.get_scipy_params(self.params))

    def cdf(self, x):
        x = self.check_x_dimension(x)
        return self.get_scipy_class().cdf(x=x, **self.get_scipy_params(self.params))

    def icdf(self, x):
        x = self.check_x_dimension(x)
        return self.get_scipy_class().ppf(x=x, **self.get_scipy_params(self.params))

    def rvs(self, nsamples):
        if not isinstance(nsamples, int) and nsamples >= 1:
            raise ValueError('Input nsamples must be an integer strictly greater than 0.')
        tmp_rvs = self.get_scipy_class().rvs(size=nsamples, **self.get_scipy_params(self.params))
        return tmp_rvs.reshape((-1, 1))

    def fit(self, x):
        x = self.check_x_dimension(x)
        return stats.lognorm.fit(data=x)

    def moments(self):
        return stats.lognorm.stats(moments='mvsk', **self.get_scipy_params(self.params))

    @staticmethod
    def get_scipy_class():
        return stats.rv_discrete

    @staticmethod
    def get_scipy_params(params):
        return {'loc': params[0], 'scale': params[1]}


class Binomial(ScipyDiscrete):
    """
    Binomial distribution, params are [n, p]
    """
    def __init__(self, params):
        if len(params) != 2:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has 2 parameters: n, p.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.binom

    @staticmethod
    def get_scipy_params(params):
        return {'n': params[0], 'p': params[1]}


class Poisson(ScipyDiscrete):
    """
    Poisson distribution, params are [mu, loc]
    """
    def __init__(self, params):
        if len(params) != 2:
            raise ValueError('The ' + self.__class__.__name__ + ' distribution has 2 parameters: mu, loc.')
        super().__init__(params=params)

    @staticmethod
    def get_scipy_class():
        return stats.uniform


########################################################################################################################
#        Multivariate Discrete Distributions
########################################################################################################################

class DistributionDiscreteND:
    """
    Define a 1-dimensional probability distribution and its associated methods:
    - pmf
    - log_pmf
    - cdf
    - rvs

    **Authors:**

    Dimitris Giovanis, Audrey Olivier, Michael D. Shields

    Last Modified: 4/23/20 by Audrey Olivier
    """
    def __init__(self):
        pass

    def __pmf(self, x):
        """
        Evaluate the probability mass function of a distribution at input points x.

        **Input:**

        :param x: Point(s) at which to evaluate the pmf.
        :type x: ndarray of shape (npoints, dimension)

        **Output/Returns:**

        :param pdf_values: Value(s) of the pmf at point(s) x.
        :type pdf_values: ndarray of shape (npoints, )
        """
        pass

    def __log_pmf(self, x):
        """
        Evaluate the logarithm of the probability mass function of a distribution at input points x.

        **Input:**

        :param x: Point(s) at which to evaluate the log-pmf.
        :type x: ndarray of shape (npoints, dimension)

        **Output/Returns:**

        :param logpdf_values: Value(s) of the log-pdf at point(s) x.
        :type logpdf_values: ndarray of shape (npoints, )
        """
        pass

    def __cdf(self, x):
        """
        Evaluate the cumulative probability function of a distribution at input points x.

        **Input:**

        :param x: Point(s) at which to evaluate the cdf.
        :type x: ndarray of shape (npoints, dimension)

        **Output/Returns:**

        :param cdf_values: Value(s) of the cdf at point(s) x.
        :type cdf_values: ndarray of shape (npoints, )
        """
        pass

    def __rvs(self, nsamples=1):
        """
        Sample iid realizations from the distribution.

        **Input:**

        :param nsamples: An integer providing the desired number of iid samples to be drawn.

                         Default: 1
        :type nsamples:  int

        **Output/Returns:**

        :return rv_s: Realizations from the distribution
        :rtype rv_s: ndarray of shape (nsamples, dimension)
        """
        pass


class Multinomial(DistributionDiscreteND):
    """
    Multinomial distribution, parameters are n (integer) and p (array-like).
    """
    def __init__(self, params):
        if len(params) != 2:
            raise ValueError('The multinomial distribution has two parameters, n and p.')
        self.params = params
        super().__init__()

    def pmf(self, x):
        pdf_val = stats.multinomial.pmf(x=x, n=self.params[0], p=self.params[1])
        return np.atleast_1d(pdf_val)

    def log_pmf(self, x):
        logpdf_val = stats.multinomial.logpmf(x=x, n=self.params[0], p=self.params[1])
        return np.atleast_1d(logpdf_val)

    def rvs(self, nsamples):
        if not (isinstance(nsamples, int) and nsamples >= 1):
            raise ValueError('Input nsamples must be an integer > 0.')
        return stats.multinomial.rvs(
            size=nsamples, n=self.params[0], p=self.params[1]).reshape((nsamples, -1))


########################################################################################################################
#        Old code
########################################################################################################################
# The supported univariate distributions are:
list_univariates = ['normal', 'uniform', 'binomial', 'beta', 'genextreme', 'chisquare', 'lognormal', 'gamma',
                    'exponential', 'cauchy', 'levy', 'logistic', 'laplace', 'maxwell', 'inverse gauss', 'pareto',
                    'rayleigh', 'truncnorm']
# The supported multivariate distributions are:
list_multivariates = ['mvnormal']
# All scipy supported distributions
list_all_scipy = list_univariates + list_multivariates

class Distribution_old:
    """
    Define a probability distribution and invoke methods of a distribution

    This is the main distribution class in UQpy. The user can define a probability distribution by providing:

    - A name that points to a univariate/multivariate distribution.
    - A list of names of univariate distributions. In that case, a multivariate  distribution is built for which all
      dimensions are independent
    - A list of names of univariate distributions and a copula. In that case a multivariate distribution is built
      using the univariate distributions for the marginal pdfs and the prescribed copula for the dependence structure.

    The Distribution class provides a number of methods as well for computing the probability density function and its
    logarithm, computing the cumulative distribution function and its inverse, generating samples of random variables
    following the distribution, fitting a distribution, and computing the moments of the distribuiton. Note that all
    methods do not exist for all distributions.

    The helper function exist_method described below indicates which methods are defined for various types of
    distributions (i.e., univariate vs. multivariate, with or without copula, user-defined).

    **Input:**

    :param dist_name: Name of the marginal distribution(s). The following distributions are available: 'normal',
                      'uniform', 'binomial', 'beta', 'genextreme', 'chisquare', 'lognormal', 'gamma', 'exponential',
                      'cauchy', 'levy', 'logistic', 'laplace', 'maxwell', 'inverse gauss', 'pareto', 'rayleigh',
                      'truncnorm', 'mvnormal'.

    :type dist_name: string or list of strings

    :param params: Parameters for the marginal distribution(s) (must be a list if distribution is multivariate).
    :type params: list or ndarray

    :param copula: Name of copula to create dependence between dimensions, used only if dist_name is a list

                   Default: None
    :type copula: str

    :param copula_params: Parameters of the copula.
    :type copula_params: list or ndarray

    **Methods:**

    Note that methods here are defined using *types.MethodType* because all of the methods listed below do not exist for
    all distributions. See the function *UQpy.Distributions.exist_method* below for further details on which methods
    are available for each distribution.

    :param self.pdf: Dynamic method that computes the probability density function (input arguments are x, params,
                     copula_params). Invoking this method executes the *UQpy.Distributions.pdf* function described
                     below.
    :type self.pdf: Callable

    :param self.cdf: Dynamic method that computes the cumulative distribution function. Invoking this method executes
                     the *UQpy.Distributions.cdf* function described below.
    :type self.cdf: Callable

    :param self.icdf: Dynamic method that computes the inverse cumulative distribution function. Invoking this method
                      executes the *UQpy.Distributions.icdf* function described below.
    :type self.icdf: Callable

    :param self.rvs: Dynamic method that generates random samples from the distribution. Invoking this method executes
                     the *UQpy.Distributions.rvs* function described below.
    :type self.rvs: Callable

    :param self.log_pdf: Dynamic method that computes the logarithm of the probability density function. Invoking this
                         method executes the *UQpy.Distributions.log_pdf* function described below.
    :type self.log_pdf: Callable

    :param self.fit: Dynamic method that estimates distribution parameters from provided data. Invoking this method
                     executes the *UQpy.Distributions.fit* function described below.
    :type self.fit: Callable

    :param self.moments: Dynamic method that calculates the first four moments of the distribution. Invoking this
                         method executes the *UQpy.Distributions.moments* function described below.
    :type self.moments: Callable

    **Authors:**

    Dimitris Giovanis, Audrey Olivier, Michael D. Shields

    Last Modified: 4/17/20 by Audrey Olivier & Michael D. Shields
    """

    def __init__(self, dist_name, copula=None, params=None, copula_params=None):

        # Check dist_name
        if isinstance(dist_name, str):
            if not (dist_name.lower() in list_all_scipy or os.path.isfile(os.path.join(dist_name + '.py'))):
                raise ValueError('dist_name should be a supported density or name of an existing .py file')
        elif isinstance(dist_name, (list, tuple)) and all(isinstance(d_, str) for d_ in dist_name):
            if not all([(d_.lower() in list_all_scipy or os.path.isfile(os.path.join(d_ + '.py')))
                        for d_ in dist_name]):
                raise ValueError('dist_name should be a list of supported densities or names of an existing .py file')
        else:
            raise TypeError('dist_name should be a (list of) string(s)')
        self.dist_name = dist_name
        self.params = None

        # Instantiate copula
        if copula is not None:
            if not isinstance(copula, str):
                raise ValueError('UQpy error: when provided, copula should be a string.')
            if isinstance(self.dist_name, str):
                raise ValueError('UQpy error: dist_name must be a list of strings to define a copula.')
            self.copula_obj = Copula_old(copula_name=copula, dist_name=self.dist_name)
            self.copula_params = None

        # Method that saves the parameters as attributes of the class if they are provided
        self.update_params(params, copula_params)

        # Other methods: you first need to check that they exist
        exist_methods = {}
        for method in ['pdf', 'log_pdf', 'cdf', 'rvs', 'icdf', 'fit', 'moments']:
            exist_methods[method] = exist_method(method=method, dist_name=self.dist_name,
                                                 has_copula=hasattr(self, 'copula'))
        if exist_methods['pdf']:
            self.pdf = MethodType(pdf, self)

        if exist_methods['cdf']:
            self.cdf = MethodType(cdf, self)

        if exist_methods['icdf']:
            self.icdf = MethodType(icdf, self)

        if exist_methods['rvs']:
            self.rvs = MethodType(rvs, self)

        if exist_methods['log_pdf']:
            self.log_pdf = MethodType(log_pdf, self)

        if exist_methods['fit']:
            self.fit = MethodType(fit, self)

        if exist_methods['moments']:
            self.moments = MethodType(moments, self)

    def update_params(self, params=None, copula_params=None):
        """
        Sets the params/copula_params attributes of the distribution.

        **Input:**

        :param params: Parameters for the marginal distribution(s) (must be a list if distribution is multivariate).
        :type params: list or ndarray

        :param copula_params: Parameters of the copula.
        :type copula_params: list or ndarray

        **Output/Returns**

        None
        """

        if params is not None:
            self.params = params
        if copula_params is not None:
            self.copula_params = copula_params


# Define the function that computes pdf
def pdf(dist_object, x, params=None, copula_params=None):
    """
    Evaluate the probability density function of a distribution dist_object at input points x.

    This is a utility function used to define the pdf method of the Distribution class. This method is called as
    dist_object.pdf(x, params, copula_params). If given, inputs params/copula_params overwrite the params/copula_params
    attributes of the dist_object.

    **Input:**

    :param dist_object: Object of the Distribution class defining the distribution
    :type dist_object: Object of *UQpy.Distributions.Distribution*

    :param x: Point(s) at which to evaluate the pdf.
    :type x: ndarray of shape (npoints, dimension)

    :param params: Parameters for the distribution
    :type params: list of lists or ndarray

    :param copula_params: Parameters of the copula
    :type copula_params: list or ndarray

    **Output/Returns:**

    :param pdf_values: Value(s) of the pdf at point(s) x.
    :type pdf_values: ndarray of shape (npoints, )
    """
    x = check_input_dims(x)
    dist_object.update_params(params, copula_params)
    if isinstance(dist_object.dist_name, str):
        return subdistribution_pdf(dist_name=dist_object.dist_name, x=x, params=dist_object.params)
    elif isinstance(dist_object.dist_name, list):
        if (x.shape[1] != len(dist_object.dist_name)) or (len(dist_object.params) != len(dist_object.dist_name)):
            raise ValueError('Inconsistent dimensions in inputs dist_name and params.')
        pdf_values = np.ones((x.shape[0],))
        for i in range(len(dist_object.dist_name)):
            pdf_values = pdf_values * subdistribution_pdf(dist_name=dist_object.dist_name[i], x=x[:, i, np.newaxis],
                                                          params=dist_object.params[i])
        if hasattr(dist_object, 'copula'):
            _, c_ = dist_object.copula_obj.evaluate_copula(x=x, dist_params=dist_object.params,
                                                       copula_params=dist_object.copula_params)
            pdf_values *= c_
        return pdf_values


# Function that computes the cdf
def cdf(dist_object, x, params=None, copula_params=None):
    """
    Evaluate the cumulative distribution function at input points x.

    This is a utility function used to define the cdf method of the Distribution class. This method is called as
    dist_object.cdf(x, params, copula_params). If given, inputs params/copula_params overwrite the params/copula_params
    attributes of the dist_object.

    **Input:**

    :param dist_object: Object of the Distribution class defining the distribution
    :type dist_object: Object of *UQpy.Distributions.Distribution*

    :param x: Point(s) at which to evaluate the cdf.
    :type x: ndarray of shape (npoints, dimension)

    :param params: Parameters for the distribution
    :type params: list of lists or ndarray

    :param copula_params: Parameters of the copula
    :type copula_params: list or ndarray

    **Output/Returns:**

    :param cdf_values: Values of the cdf at points x.
    :type cdf_values: ndarray of shape (npoints, )
    """
    x = check_input_dims(x)
    dist_object.update_params(params, copula_params)
    if isinstance(dist_object.dist_name, str):
        return subdistribution_cdf(dist_name=dist_object.dist_name, x=x, params=dist_object.params)
    elif isinstance(dist_object.dist_name, list):
        if (x.shape[1] != len(dist_object.dist_name)) or (len(params) != len(dist_object.dist_name)):
            raise ValueError('Inconsistent dimensions in inputs dist_name and params.')
        if not hasattr(dist_object, 'copula'):
            cdfs = np.zeros_like(x)
            for i in range(len(dist_object.dist_name)):
                cdfs[:, i] = subdistribution_cdf(dist_name=dist_object.dist_name[i], x=x[:, i, np.newaxis],
                                                 params=dist_object.params[i])
            return np.prod(cdfs, axis=1)
        else:
            cdf_values, _ = dist_object.copula.evaluate_copula(x=x, dist_params=params, copula_params=copula_params)
            return cdf_values


# Method that computes the icdf
def icdf(dist_object, x, params=None):
    """
    Evaluate the inverse distribution function at inputs points x (only for univariate distributions).

    This is a utility function used to define the icdf method of the Distribution class. This method is called as
    dist_object.icdf(x, params). If given, input params overwrites the params attributes of the dist_object.

    **Input:**

    :param dist_object: Object of the Distribution class defining the distribution
    :type dist_object: Object of *UQpy.Distributions.Distribution*

    :param x: Point(s) where to evaluate the icdf.
    :type x: ndarray of shape (npoints, 1)

    :param params: Parameters for the distribution
    :type params: list or ndarray

    **Output/Returns:**

    :param icdf_values: Values of the icdf at points x.
    :type icdf_values: ndarray of shape (npoints, )
    """
    x = check_input_dims(x)
    dist_object.update_params(params, copula_params=None)
    if isinstance(dist_object.dist_name, str):
        return subdistribution_icdf(dist_name=dist_object.dist_name, x=x, params=dist_object.params)
    else:
        raise AttributeError('Method icdf not defined for multivariate distributions.')


# Method that generates RVs
def rvs(dist_object, nsamples=1, params=None):
    """
    Sample iid realizations from the distribution - does not support distributions with copula.

    This is a utility function used to define the rvs method of the Distribution class. This method is called as
    dist_object.rvs(x, params). If given, input params overwrites the params attributes of the dist_object.

    **Input:**

    :param dist_object: Object of the Distribution class defining the distribution
    :type dist_object: Object of *UQpy.Distributions.Distribution*

    :param nsamples: An integer providing the desired number of iid samples to be drawn.

                     Default: 1
    :type nsamples:  int

    :param params: Parameters for the distribution
    :type params: list or ndarray

    **Output:**

    :return rv_s: Realizations from the distribution
    :rtype rv_s: ndarray of shape (nsamples, dimension)
    """
    dist_object.update_params(params, copula_params=None)
    if isinstance(dist_object.dist_name, str):
        return subdistribution_rvs(dist_name=dist_object.dist_name, nsamples=nsamples, params=dist_object.params)
    elif isinstance(dist_object.dist_name, list):
        if len(dist_object.params) != len(dist_object.dist_name):
            raise ValueError('UQpy error: Inconsistent dimensions')
        if not hasattr(dist_object, 'copula'):
            rv_s = np.zeros((nsamples, len(dist_object.dist_name)))
            for i in range(len(dist_object.dist_name)):
                rv_s[:, i] = subdistribution_rvs(dist_name=dist_object.dist_name[i], nsamples=nsamples,
                                                 params=dist_object.params[i])[:, 0]
            return rv_s
        else:
            raise AttributeError('Method rvs not defined for distributions with copula.')


# Define the function that computes the log pdf
def log_pdf(dist_object, x, params=None, copula_params=None):
    """
    Evaluate the logarithm of the probability density function of a distribution at input points x.

    This is a utility function used to define the log_pdf method of the Distribution class. This method is called as
    dist_object.log_pdf(x, params, copula_params). If given, inputs params/copula_params overwrite the
    params/copula_params attributes of the dist_object.

    **Input:**

    :param dist_object: Object of the Distribution class defining the distribution
    :type dist_object: Object of *UQpy.Distributions.Distribution*

    :param x: Points where to estimate the log-pdf.
    :type x: 2D ndarray (npoints, dimension)

    :param params: Parameters of the distribution.
    :type params: list or ndarray

    :param copula_params: Parameters of the copula.
    :type copula_params: list or ndarray

    **Output/Returns:**

    :param log_pdf_values: Values of the log-pdf evaluated at points x.
    :type log_pdf_values: ndarray of shape (npoints, )
    """
    x = check_input_dims(x)
    dist_object.update_params(params, copula_params)
    if isinstance(dist_object.dist_name, str):
        return subdistribution_log_pdf(dist_name=dist_object.dist_name, x=x, params=dist_object.params)
    elif isinstance(dist_object.dist_name, list):
        if (x.shape[1] != len(dist_object.dist_name)) or (len(dist_object.params) != len(dist_object.dist_name)):
            raise ValueError('Inconsistent dimensions in inputs dist_name and params.')
        log_pdf_values = np.zeros((x.shape[0],))
        for i in range(len(dist_object.dist_name)):
            log_pdf_values = log_pdf_values + subdistribution_log_pdf(dist_name=dist_object.dist_name[i],
                                                                      x=x[:, i, np.newaxis],
                                                                      params=dist_object.params[i])
        if hasattr(dist_object, 'copula'):
            _, c_ = dist_object.copula_obj.evaluate_copula(
                x=x, dist_params=dist_object.params, copula_params=dist_object.copula_params)
            log_pdf_values += np.log(c_)
        return log_pdf_values


def fit(dist_object, x):
    """
    Compute the MLE parameters of a distribution from data x - does not support distributions with copula.

    This is a utility function used to define the fit method of the Distribution class. This method is called as
    dist_object.fit(x).

    **Input:**

    :param dist_object: Object of the Distribution class defining the distribution
    :type dist_object: Object of *UQpy.Distributions.Distribution*

    :param x: Vector of data x, contains iid samples from the distribution
    :type x: ndarray of shape (nsamples, dimension)

    **Output/Returns:**

    :param params_fit: MLE parameters.
    :type params_fit: ndarray
    """
    x = check_input_dims(x)
    if isinstance(dist_object.dist_name, str):
        return subdistribution_fit(dist_name=dist_object.dist_name, x=x)
    elif isinstance(dist_object.dist_name, list):
        if x.shape[1] != len(dist_object.dist_name):
            raise ValueError('Inconsistent dimensions in inputs dist_name and x.')
        if not hasattr(dist_object, 'copula'):
            params_fit = []
            for i in range(len(dist_object.dist_name)):
                params_fit.append(subdistribution_fit(dist_name=dist_object.dist_name[i], x=x[:, i, np.newaxis]))
            return params_fit
        else:
            raise AttributeError('Method fit not defined for distributions with copula.')


# Method that computes moments
def moments(dist_object, params=None):
    """
    Compute marginal moments (mean, variance, skewness, kurtosis). Does not support distributions with copula.

    This is a utility function used to define the moments method of the Distribution class. This method is called as
    dist_object.moments(params). If given, input params overwrites the params attributes of the dist_object.

    **Input:**

    :param dist_object: Object of the Distribution class defining the distribution
    :type dist_object: Object of *UQpy.Distributions.Distribution*

    :param params: Parameters of the distribution
    :type params: list or ndarray

    **Output/Returns:**

    :param mean: Mean value(s).
    :type mean: list

    :param var: Variance(s).
    :type var: list

    :param skew: Skewness value(s).
    :type skew: list

    :param kurt: Kurtosis value(s).
    :type kurt: list
     """
    dist_object.update_params(params, copula_params=None)
    if isinstance(dist_object.dist_name, str):
        return subdistribution_moments(dist_name=dist_object.dist_name, params=dist_object.params)
    elif isinstance(dist_object.dist_name, list):
        if len(dist_object.params) != len(dist_object.dist_name):
            raise ValueError('UQpy error: Inconsistent dimensions')
        if not hasattr(dist_object, 'copula'):
            mean, var, skew, kurt = [0] * len(dist_object.dist_name), [0] * len(dist_object.dist_name), [0] * len(
                dist_object.dist_name), \
                                    [0] * len(dist_object.dist_name),
            for i in range(len(dist_object.dist_name)):
                mean[i], var[i], skew[i], kurt[i] = subdistribution_moments(dist_name=dist_object.dist_name[i],
                                                                            params=dist_object.params[i])
            return mean, var, skew, kurt
        else:
            raise AttributeError('Method moments not defined for distributions with copula.')


class Copula_old:
    """
    Define a copula for a multivariate distribution whose dependence structure is defined with a copula.

    This class is used in support of the main Distribution class. The following copula are supported: Gumbel.

    **Input:**

    :param copula_name: Name of copula.
    :type copula_name: string

    :param dist_name: Names of the marginal distributions.
    :type dist_name: list of strings
    """

    def __init__(self, copula_name, dist_name):

        self.copula_name = copula_name
        self.dist_name = dist_name

    def evaluate_copula(self, x, dist_params, copula_params):
        """
        Compute the copula cdf c and copula density c_ necessary to evaluate the cdf and pdf, respectively, of the
        associated multivariate distribution.

        **Input:**

        :param x: Points at which to evaluate the copula cdf and pdf.
        :type x: ndarray of shape (npoints, dimension)

        :param dist_params: Parameters of the marginal distributions.
        :type dist_params: list of lists or ndarray

        :param copula_params: Parameter of the copula.
        :type copula_params: list or ndarray

        **Output/Returns**

        :param c: Copula cdf
        :type c: ndarray

        :param c\_: Copula pdf
        :type c\_: ndarray
        """
        if self.copula_name.lower() == 'gumbel':
            if x.shape[1] > 2:
                raise ValueError('Maximum dimension for the Gumbel Copula is 2.')
            if not isinstance(copula_params, (list, np.ndarray)):
                copula_params = [copula_params]
            if copula_params[0] < 1:
                raise ValueError('The parameter for Gumbel copula must be defined in [1, +oo)')

            uu = np.zeros_like(x)
            for i in range(uu.shape[1]):
                uu[:, i] = subdistribution_cdf(dist_name=self.dist_name[i], x=x[:, i, np.newaxis],
                                               params=dist_params[i])
            if copula_params[0] == 1:
                return np.prod(uu, axis=1), np.ones(x.shape[0])
            else:
                u = uu[:, 0]
                v = uu[:, 1]
                c = np.exp(-((-np.log(u)) ** copula_params[0]+(-np.log(v)) ** copula_params[0]) **
                            (1/copula_params[0]))

                c_ = c * 1/u*1/v*((-np.log(u)) ** copula_params[0]+(-np.log(v)) ** copula_params[0]) ** \
                    (-2 + 2/copula_params[0]) * (np.log(u) * np.log(v)) ** (copula_params[0]-1) *\
                    (1 + (copula_params[0] - 1) * ((-np.log(u)) ** copula_params[0] +
                                                   (-np.log(v)) ** copula_params[0]) ** (-1/copula_params[0]))
                return c, c_
        else:
            raise NotImplementedError('Copula type not supported!')


def exist_method(method, dist_name, has_copula):
    """
    Check whether a method exists for a given distribution.

    In particular:

    - All methods exist for univariate scipy distributions,
    - Multivariate scipy distributions have pdf, logpdf, cdf and rvs,
    - icdf method does not exist for any multivariate distribution,
    - rvs, fit and moments method do not exist for multivariate distributions with copula.
    - For any multivariate distribution with independent marginals, a method exists if it exists for all its marginals.
    - For custom distributions, only methods provided within the corresponding .py file exist.

    **Inputs:**

    :param method: Name of the method to be checked (pdf, cdf, icdf, rvs, log_pdf, moments, fit)
    :type method: str

    :param dist_name: Name of the marginal distribution(s)
    :type dist_name: str or list of str

    :param has_copula: indicates whether a copula exists
    :type has_copula: bool

    **Output/Returns:**

    :param method_exist: Indicates whether the method exists for this distribution
    :type method_exist: bool
    """
    if isinstance(dist_name, str):    # check the subdistribution
        return subdistribution_exist_method(dist_name=dist_name, method=method)
    elif isinstance(dist_name, (list, tuple)):    # Check all the subdistributions
        if method == 'icdf':
            return False
        # distributions with copula do not have rvs, fit, moments
        if has_copula and (method in ['moments', 'fit', 'rvs']):
            return False
        # method exist if all subdistributions have the corresponding method
        if all([subdistribution_exist_method(dist_name=n, method=method) for n in dist_name]):
            return True
    return False


# The following functions are helper functions for subdistributions, i.e., distributions where dist_name
# is only a string

def subdistribution_exist_method(dist_name, method):
    """
    Check whether a method exists for a given sub-distribution (i.e., a distribution defined by a single string,
    univariate marginal or custom file).

    This is a helper function, used within the main Distribution class and exist_method function.

    **Inputs:**

    :param method: name of the method to be checked (pdf, cdf, icdf, rvs, log_pdf, moments, fit)
    :type method: str

    :param dist_name: name of the sub-distribution
    :type dist_name: str

    **Output/Returns:**

    :param method_exist: indicates whether the method exist for this distribution
    :type method_exist: bool

    """
    if dist_name.lower() in list_univariates:
        return True
    # Multivariate scipy distributions have pdf, logpdf, cdf and rvs
    elif dist_name.lower() in list_multivariates:
        if method in ['pdf', 'log_pdf', 'cdf', 'rvs']:
            return True
        else:
            return False
    # User defined distributions: check !
    else:
        custom_dist = importlib.import_module(dist_name)
        return hasattr(custom_dist, method)


def subdistribution_pdf(dist_name, x, params):
    """
    Evaluate pdf for sub-distribution (i.e., distribution defined by a single string, univariate marginal or custom).

    This is a helper function, used within the main Distribution class.

    **Inputs:**

    :param dist_name: name of the sub-distribution
    :type dist_name: str

    :param x: points where to evaluate the pdf
    :type x: ndarray

    :param params: parameters of the sub-distribution
    :type params: list or ndarray

    **Output/Returns:**

    :param pdf_val: values of the pdf evaluated at points in x
    :type pdf_val: ndarray of shape (npoints,)

    """
    if dist_name.lower() in list_univariates:
        d, kwargs = scipy_distributions(dist_name=dist_name, params=params)
        val = d.pdf(x[:, 0], **kwargs)
    elif dist_name.lower() in list_multivariates:
        d, kwargs = scipy_distributions(dist_name=dist_name, params=params)
        val = d.pdf(x, **kwargs)
    # Otherwise it must be a file
    else:
        custom_dist = importlib.import_module(dist_name)
        tmp = getattr(custom_dist, 'pdf', None)
        val = tmp(x, params=params)
    if isinstance(val, (float, int)):
        val = np.array([val])
    return val


def subdistribution_cdf(dist_name, x, params):
    """
    Evaluate cdf for sub-distribution (i.e., distribution defined by a single string, univariate marginal or custom).

    This is a helper function, used within the main Distribution class.

    **Inputs:**

    :param dist_name: name of the sub-distribution
    :type dist_name: str

    :param x: points where to evaluate the cdf
    :type x: ndarray

    :param params: parameters of the sub-distribution
    :type params: list or ndarray

    **Output/Returns:**

    :param cdf_val: values of the cdf evaluated at points in x
    :type cdf_val: ndarray of shape (npoints,)

    """
    # If it is a supported scipy distribution:
    if dist_name.lower() in list_univariates:
        d, kwargs = scipy_distributions(dist_name=dist_name, params=params)
        val = d.cdf(x=x[:, 0], **kwargs)
    elif dist_name.lower() in list_multivariates:
        d, kwargs = scipy_distributions(dist_name=dist_name, params=params)
        val = d.cdf(x=x, **kwargs)
    # Otherwise it must be a file
    else:
        custom_dist = importlib.import_module(dist_name)
        tmp = getattr(custom_dist, 'cdf')
        val = tmp(x, params)
    if isinstance(val, (float, int)):
        val = np.array([val])
    return val


def subdistribution_icdf(dist_name, x, params):
    """
    Evaluate inverse cdf for sub-distribution (i.e., distribution defined by a single string, univariate marginal
    or custom).

    This is a helper function, used within the main Distribution class.

    **Inputs:**

    :param dist_name: name of the sub-distribution
    :type dist_name: str

    :param x: points where to evaluate the icdf
    :type x: ndarray

    :param params: parameters of the sub-distribution
    :type params: list or ndarray

    **Output/Returns:**

    :param icdf_val: values of the icdf evaluated at points in x
    :type icdf_val: ndarray of shape (npoints,)

    """
    # If it is a supported scipy distribution:
    if dist_name.lower() in list_univariates:
        d, kwargs = scipy_distributions(dist_name=dist_name, params=params)
        val = d.ppf(x[:, 0], **kwargs)
    elif dist_name.lower() in list_multivariates:
        d, kwargs = scipy_distributions(dist_name=dist_name, params=params)
        val = d.ppf(x, **kwargs)
    # Otherwise it must be a file
    else:
        custom_dist = importlib.import_module(dist_name)
        tmp = getattr(custom_dist, 'icdf')
        val = tmp(x, params)
    if isinstance(val, (float, int)):
        val = np.array([val])
    return val


def subdistribution_rvs(dist_name, nsamples, params):
    """
    Sample realizations from sub-distribution (i.e., distribution defined by a single string, marginal or custom).

    This is a helper function, used within the main Distribution class.

    **Inputs:**

    :param dist_name: name of the sub-distribution
    :type dist_name: str

    :param nsamples: number of realizations to sample
    :type nsamples: int

    :param params: parameters of the sub-distribution
    :type params: list or ndarray

    **Output/Returns:**

    :param samples: realizations of the sub-distribution
    :type samples: ndarray

    """
    # If it is a supported scipy distribution:
    if dist_name.lower() in (list_univariates + list_multivariates):
        d, kwargs = scipy_distributions(dist_name=dist_name, params=params)
        rv_s = d.rvs(size=nsamples, **kwargs)
    # Otherwise it must be a file
    else:
        custom_dist = importlib.import_module(dist_name)
        tmp = getattr(custom_dist, 'rvs')
        rv_s = tmp(nsamples=nsamples, params=params)
    if isinstance(rv_s, (float, int)):
        return np.array([[rv_s]])    # one sample in a 1d space
    if len(rv_s.shape) == 1:
        if nsamples == 1:
            return rv_s[np.newaxis, :]    # one sample in a d-dimensional space
        else:
            return rv_s[:, np.newaxis]    # several samples in a one-dimensional space
    return rv_s


def subdistribution_log_pdf(dist_name, x, params):
    """
    Evaluate logpdf for sub-distribution (i.e., distribution defined by a single string, univariate marginal or custom).

    This is a helper function, used within the main Distribution class.

    **Inputs:**

    :param dist_name: name of the sub-distribution
    :type dist_name: str

    :param x: points where to evaluate the log-pdf
    :type x: ndarray

    :param params: parameters of the sub-distribution
    :type params: list or ndarray

    **Output/Returns:**

    :param logpdf_val: values of the log-pdf evaluated at points in x
    :type logpdf_val: ndarray of shape (npoints,)

    """
    # If it is a supported scipy distribution:
    if dist_name.lower() in list_univariates:
        d, kwargs = scipy_distributions(dist_name=dist_name, params=params)
        val = d.logpdf(x[:, 0], **kwargs)
    elif dist_name.lower() in list_multivariates:
        d, kwargs = scipy_distributions(dist_name=dist_name, params=params)
        val = d.logpdf(x, **kwargs)
    # Otherwise it must be a file
    else:
        custom_dist = importlib.import_module(dist_name)
        tmp = getattr(custom_dist, 'log_pdf')
        val = tmp(x, params)
    if isinstance(val, (float, int)):
        val = np.array([val])
    return val


def subdistribution_fit(dist_name, x):
    """
    Fit parameters of sub-distribution (i.e., distribution defined by a single string, univariate marginal or custom).

    This is a helper function, used within the main Distribution class.

    **Inputs:**

    :param dist_name: name of the sub-distribution
    :type dist_name: str

    :param x: data used for fitting
    :type x: ndarray

    **Output/Returns:**

    :param mle_params: fitted parameters
    :type mle_params: ndarray

    """
    # If it is a supported scipy distribution:
    if dist_name.lower() in list_univariates:
        d, kwargs = scipy_distributions(dist_name=dist_name)
        return d.fit(x[:, 0])
    elif dist_name.lower() in list_multivariates:
        d, kwargs = scipy_distributions(dist_name=dist_name)
        return d.fit(x)
    # Otherwise it must be a file
    else:
        custom_dist = importlib.import_module(dist_name)
        tmp = getattr(custom_dist, 'fit')
        return tmp(x)


def subdistribution_moments(dist_name, params):
    """
    Compute moments of sub-distribution (i.e., distribution defined by a single string, univariate marginal or custom).

    This is a helper function, used within the main Distribution class.

    **Inputs:**

    :param dist_name: name of the sub-distribution
    :type dist_name: str

    :param params: parameters of the sub-distribution
    :type params: list or ndarray

    **Output/Returns:**

    :param moments: moments of sub-distribution (mean, var, skewness, kurtosis)
    :type moments: ndarray of shape (4,)

    """
    # If it is a supported scipy distribution:
    if dist_name.lower() in list_univariates:
        y = [np.nan, np.nan, np.nan, np.nan]
        d, kwargs = scipy_distributions(dist_name=dist_name, params=params)
        mean, var, skew, kurt = d.stats(moments='mvsk', **kwargs)
        y[0] = mean
        y[1] = var
        y[2] = skew
        y[3] = kurt
        return np.array(y)
    # Otherwise it must be a file
    else:
        custom_dist = importlib.import_module(dist_name)
        tmp = getattr(custom_dist, 'moments')
        return tmp(params=params)


def scipy_distributions(dist_name, params=None):
    """
    Create a scipy distribution object and map argument params to the scipy parameters.

    This is a helper function, used within the main Distribution class that serves to translate UQpy distribution
    parameters to scipy distribution parameters in order to leverage scipy distribution objects.

    **Inputs:**

    :param dist_name: name of the sub-distribution
    :type dist_name: str

    :param params: parameters of the sub-distribution
    :type params: list or ndarray

    **Output/Returns:**

    :param dist: scipy.stats distribution object
    :type dist: object

    :param params_dict: dictionary that maps the scipy parameters (scale, loc...) to elements of vector params
    :type params_dict: dict

    """
    kwargs = {}
    if params is not None:
        kwargs = {'loc': params[0], 'scale': params[1]}

    if dist_name.lower() == 'normal' or dist_name.lower() == 'gaussian':
        return stats.norm, kwargs

    elif dist_name.lower() == 'uniform':
        return stats.uniform, kwargs

    elif dist_name.lower() == 'binomial':
        if params is not None:
            kwargs = {'n': params[0], 'p': params[1]}
        return stats.binom, kwargs

    elif dist_name.lower() == 'beta':
        if params is not None:
            kwargs = {'a': params[0], 'b': params[1]}
        return stats.beta, kwargs

    elif dist_name.lower() == 'genextreme':
        if params is not None:
            kwargs = {'c': params[0], 'loc': params[0], 'scale': params[1]}
        return stats.genextreme, kwargs

    elif dist_name.lower() == 'chisquare':
        if params is not None:
            kwargs = {'df': params[0], 'loc': params[1], 'scale': params[2]}
        return stats.chi2, kwargs

    elif dist_name.lower() == 'lognormal':
        if params is not None:
            kwargs = {'s': params[0], 'loc': params[1], 'scale': params[2]}
        return stats.lognorm, kwargs

    elif dist_name.lower() == 'gamma':
        if params is not None:
            kwargs = {'a': params[0], 'loc': params[1], 'scale': params[2]}
        return stats.gamma, kwargs

    elif dist_name.lower() == 'exponential':
        return stats.expon, kwargs

    elif dist_name.lower() == 'cauchy':
        return stats.cauchy, kwargs

    elif dist_name.lower() == 'inverse gauss':
        if params is not None:
            kwargs = {'mu': params[0], 'loc': params[1], 'scale': params[2]}
        return stats.invgauss, kwargs

    elif dist_name.lower() == 'logistic':
        return stats.logistic, kwargs

    elif dist_name.lower() == 'pareto':
        if params is not None:
            kwargs = {'b': params[0], 'loc': params[1], 'scale': params[2]}
        return stats.pareto, kwargs

    elif dist_name.lower() == 'rayleigh':
        return stats.rayleigh, kwargs

    elif dist_name.lower() == 'levy':
        return stats.levy, kwargs

    elif dist_name.lower() == 'laplace':
        return stats.laplace, kwargs

    elif dist_name.lower() == 'maxwell':
        return stats.maxwell, kwargs

    elif dist_name.lower() == 'truncnorm':
        if params is not None:
            kwargs = {'a': params[0], 'b': params[1], 'loc': params[2], 'scale': params[3]}
        return stats.truncnorm, kwargs

    elif dist_name.lower() == 'mvnormal':
        if params is not None:
            kwargs = {'mean': params[0], 'cov': params[1]}
        return stats.multivariate_normal, kwargs
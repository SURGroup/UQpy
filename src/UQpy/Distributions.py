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
This module contains functionality for all the probability distributions supported in ``UQpy``. The ``Distributions``
module is  used  to  define  probability  distribution  objects.   These  objects  possess various  methods  that  allow
the user  to:  compute  the  probability  density  function ``pdf``, the cumulative  distribution  function ``cdf``, the
logarithm of the pdf ``log_pdf``, return the moments, draw independent samples ``rvs`` and fit the parameters of the
model from data ``fit``.  Each of these methods are designed to be consistent with the ``scipy.stats`` package to ensure
compatibility with common Python standards.

The module currently contains the following classes:

- ``Distribution``: Parent class for all distribution classes supported in ``UQpy``.
- ``DistributionContinuous1D``: Defines a 1-dimensional continuous probability distribution in ``UQpy``.
- ``DistributionDiscrete1D``: Defines a 1-dimensional discrete probability distribution in ``UQpy``.
- ``DistributionND``: Defines a multivariate probability distribution in ``UQpy``.
- ``Copula``: Defines a copula for modeling dependence in multivariate distributions.

Example Usage:

To instantiate a *univariate lognormal* distribution::

    >>> from UQpy.Distributions import Lognormal
    >>> dist = Lognormal(s=1, loc=0, scale=np.exp(5))
    >>> print(dist)
        {'s': 1, 'loc': 0, 'scale': 148.4131591025766}

To create values from its *probability density function (pdf)*::

    >>> x = np.linspace(50, 100, 3).reshape((-1, 1))
        # Input x must be a 2D array (nsamples, dimension)
    >>> dist.pdf(x).round(4)
        array([0.0044, 0.0042, 0.0037])


The *mean*, *standard deviation*, *skewness*, and *kurtosis* of the distribution are::

    >>> moments_list = ['mean', 'variance', 'skewness', 'kurtosis']
    >>> m = dist.moments()
    >>> print('Moments with inherited parameters:')
    >>> for i, moment in enumerate(moments_list):
    >>>     print(moment+' = {0:.2f}'.format(m[i]))
        Moments with inherited parameters:
        mean = 344.69
        variance = 102880.65
        skewness = 6.18
        kurtosis = 110.94

Notice that, when calling the ``moments`` method, the parameters are inherited from the class if they are not specified.
If  the parameters are specified, then they overwrite the parameters in the defined class.

To generate 5 random samples from the lognormal distribution.

    >>> np.random.seed(123) # To reproduce results
    >>> y = dist.rvs(nsamples=5).round(3)
        array([[ 50.117], [402.359], [196.956], [ 32.908], [ 83.213]])

Notice that when calling the rvs method, the number of samples must be specified.  Again, the parameters can be
inherited from the Distribution object or overwritten.

In addition, we can update its parameters with the need to instantiate a new object::

    >>> dist.update_parameters(params=[1, 100, np.exp(5)])

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


class Distribution:
    """
    A parent class to all probability distributions

    **cdf(x)**
             Evaluate the cumulative probability function of a distribution.

        **Input:**

                x (np.ndarray):
                            Point(s) at which to evaluate the *cdf*. ``x.shape`` must
                            be of shape ``(npoints,)`` or ``(npoints, 1)``.

        **Output/Returns:**

                (ndarray):
                        Evaluated distribution function values.


    """
    def __init__(self):
        pass

    def __cdf(self, x):
        """
        Cumulative distribution function.

        Evaluate the cumulative probability function of a distribution.

        **Input:**

                x (np.ndarray):
                            Point(s) at which to evaluate the *cdf*. ``x.shape`` must
                            be of shape ``(npoints,)`` or ``(npoints, 1)``.

        **Output/Returns:**

                (ndarray):
                        Evaluated distribution function values.

        """
        pass

    def __pdf(self, x):
        """
        Probability density function (Continuous).

        Evaluate the probability density function of a continuous distribution.

        **Input:**

                x (np.ndarray):
                            Point(s) at which to evaluate the *pdf*. ``x.shape`` must
                            be of shape ``(npoints,)`` or ``(npoints, 1)``.

        **Output/Returns:**

                (ndarray):
                        Evaluated  density function values.

        """
        pass

    def __log_pdf(self, x):
        """
        Logarithm of the probability density function (Continuous).

        Evaluate the logarithm of the probability density function of a continuous distribution.

        **Input:**

                x (np.ndarray):
                            Point(s) at which to evaluate the *logpdf*. ``x.shape`` must
                            be of shape ``(npoints,)`` or ``(npoints, 1)``.

        **Output/Returns:**

                (ndarray):
                        Evaluated  logarithmnic density function values.
        """
        pass

    def __icdf(self, x):
        """
        Inverse cumulative distribution function.

        Evaluate the inverse of the cumulative probability function of a distribution.

        **Input:**

                x (np.ndarray):
                            Point(s) at which to evaluate the *icdf*. ``x.shape`` must
                            be of shape ``(npoints,)`` or ``(npoints, 1)``.

        **Output/Returns:**

                (ndarray):
                        Evaluated  logarithmnic density function values.
        """
        pass

    def __rvs(self, nsamples=1):
        """
        Draw pseudo-random samples.

        Generate independent and identical distributed (iid) realizations from the distribution.

        **Input:**

                nsamples (integer):
                            Number of iid samples to draw from the distribution.

                            ``Default: 1``

        **Output/Returns:**

                (ndarray):
                        Realizations from the distribution with shape ``(nsamples, 1)``.

        """
        pass

    def __mle(self, x):
        """
        Maximum Likelihood Estimation (MLE).

        Compute the MLE parameters of a distribution from data.

        **Input:**

                x (ndarray):
                           Array of iid samples from the distribution with shape ``(npoints,)`` or ``(npoints, 1)``.

        **Output/Returns:**

                (ndarray):
                        MLE parameters.
        """
        pass

    def __moments(self):
        """
        Moments of a distribution.

        Compute the moments (mean, variance, skewness, kurtosis).

        **Input:**

                None

        **Output/Returns:**

                (ndarray):``mean`` (float): mean value, ``var`` (float):  variance,
                          ``skew`` (float): skewness, ``kurt`` (float): kyrtosis.

        """
        pass

    def __pmf(self, x):
        """
        Probability mass function (Discrete).

        Evaluate the probability mass function of a discrete distribution.

        **Input:**

                x (np.ndarray):
                            Point(s) at which to evaluate the *pmf*. ``x.shape`` must
                            be of shape ``(npoints,)`` or ``(npoints, 1)``.

        **Output/Returns:**

                (ndarray):
                        Evaluated  probability mass function function values of shape ``(npoints,)``.

        """
        pass

    def __log_pmf(self, x):
        """
        Logarithm of probability mass function (Discrete).

        Evaluate the logarithm of the probability mass function of a discrete distribution.

        **Input:**

                x (np.ndarray):
                            Point(s) at which to evaluate the *logpmf*. ``x.shape`` must
                            be of shape ``(npoints,)`` or ``(npoints, 1)``.

        **Output/Returns:**

                (ndarray):
                        Evaluated  logarithmnic mass function values.
        """
        pass

    @staticmethod
    def check_x_dimension(x):
        """
        Check the dimension.

        Help function that check the dimension of x - must be an ndarray of shape ``(npoints,)`` or ``(npoints, 1)``.
        """
        x = np.atleast_1d(x)
        print(x.shape)
        if len(x.shape) > 2 or (len(x.shape) == 2 and x.shape[1] != 1):
            raise ValueError('Wrong dimension in x.')
        return x.reshape((-1,))

    @staticmethod
    def check_x_dimension_mv(x, d=None):
        """
        Check the dimension of input x - must be an ndarray of shape (npoints, d)
        """
        x = np.array(x)
        if len(x.shape) != 2:
            raise ValueError('Wrong dimension in x.')
        if (d is not None) and (x.shape[1] != d):
            raise ValueError('Wrong dimension in x.')
        return x


class DistributionContinuous1D(Distribution):
    """
    Define a 1-dimensional continuous probability distribution and its associated methods.

    """
    def __init__(self, **kwargs):
        super().__init__()
        self.parameters = kwargs


########################################################################################################################
#        Univariate Continuous Distributions
########################################################################################################################


class ScipyContinuous(DistributionContinuous1D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scipy_name = stats.rv_continuous
        self.ordered_params = ['loc', 'scale']

    def pdf(self, x):
        x = self.check_x_dimension(x)
        return self.scipy_name.pdf(x=x, **self.parameters)

    def log_pdf(self, x):
        x = self.check_x_dimension(x)
        return self.scipy_name.logpdf(x=x, **self.parameters)

    def cdf(self, x):
        x = self.check_x_dimension(x)
        return self.scipy_name.cdf(x=x, **self.parameters)

    def icdf(self, x):
        x = self.check_x_dimension(x)
        return self.scipy_name.ppf(x=x, **self.parameters)

    def rvs(self, nsamples):
        if not isinstance(nsamples, int) and nsamples >= 1:
            raise ValueError('Input nsamples must be an integer strictly greater than 0.')
        tmp_rvs = self.scipy_name.rvs(size=nsamples, **self.parameters)
        return tmp_rvs.reshape((-1, 1))

    def moments(self):
        return self.scipy_name.stats(moments='mvsk', **self.parameters)

    def update_parameters(self, params, fixed_params=None):
        if fixed_params is None:
            fixed_params = {}
        if (len(np.array(params).shape) != 1) or (len(params) + len(fixed_params) != len(self.parameters)):
            raise ValueError('Incorrect number of parameters.')
        cnt_params = 0
        for key in self.ordered_params:
            if key in fixed_params.keys():
                self.parameters[key] = fixed_params[key]
            else:
                self.parameters[key] = params[cnt_params]
                cnt_params += 1


class Normal(ScipyContinuous):
    """
    Normal distribution.

    Params: [loc, scale]

    Examples:

    >>> from UQpy.Distributions import Normal
    >>> dist = Normal(loc=2., scale=1.)
    >>> x = np.linspace(0., 1.0, 3).reshape(-1, 1)
    >>> dist.pdf(x).round(3)
        array([0.054, 0.13 , 0.242])
    >>> dist.cdf(x).round(3)
        array([0.023, 0.067, 0.159])
    """
    def __init__(self, loc=0., scale=1.):
        super().__init__(loc=loc, scale=scale)
        self.scipy_name = stats.norm


class Uniform(ScipyContinuous):
    """
    Uniform distribution, params are [loc, scale]
    """
    def __init__(self, loc=0., scale=1.):
        super().__init__(loc=loc, scale=scale)
        self.scipy_name = stats.uniform


class Beta(ScipyContinuous):
    """
    Beta distribution, params are [a, b, loc, scale]
    """
    def __init__(self, a, b, loc=0., scale=1.):
        super().__init__(a=a, b=b, loc=loc, scale=scale)
        self.scipy_name = stats.beta
        self.ordered_params = ['a', 'b', 'loc', 'scale']


class Genextreme(ScipyContinuous):
    """
    Genextreme distribution, params are [c, loc, scale]
    """
    def __init__(self, c, loc=0., scale=1.):
        super().__init__(c=c, loc=loc, scale=scale)
        self.scipy_name = stats.genextreme
        self.ordered_params = ['c', 'loc', 'scale']


class Chisquare(ScipyContinuous):
    """
    Chisquare distribution, params are [df, loc, scale]
    """
    def __init__(self, df, loc=0., scale=1):
        super().__init__(df=df, loc=loc, scale=scale)
        self.scipy_name = stats.chi2
        self.ordered_params = ['df', 'loc', 'scale']


class Lognormal(ScipyContinuous):
    """
    Lognormal distribution, params are [s, loc, scale]
    """
    def __init__(self, s, loc=0., scale=1.):
        super().__init__(s=s, loc=loc, scale=scale)
        self.scipy_name = stats.lognorm
        self.ordered_params = ['s', 'loc', 'scale']


class Gamma(ScipyContinuous):
    """
    Gamma distribution, params are [a, loc, scale]
    """
    def __init__(self, a, loc=0., scale=1.):
        super().__init__(a=a, loc=loc, scale=scale)
        self.scipy_name = stats.gamma
        self.ordered_params = ['a', 'loc', 'scale']


class Exponential(ScipyContinuous):
    """
    Exponential distribution, params are [loc, scale]
    """
    def __init__(self, loc=0., scale=1.):
        super().__init__(loc=loc, scale=scale)
        self.scipy_name = stats.expon


class Cauchy(ScipyContinuous):
    """
    Cauchy distribution, params are [loc, scale]
    """
    def __init__(self, loc=0., scale=1.):
        super().__init__(ploc=loc, scale=scale)
        self.scipy_name = stats.cauchy


class InvGauss(ScipyContinuous):
    """
    Inverse Gauss distribution, params are [mu, loc, scale]
    """
    def __init__(self, mu, loc=0., scale=1.):
        super().__init__(mu=mu, loc=loc, scale=scale)
        self.scipy_name = stats.invgauss
        self.ordered_params = ['mu', 'loc', 'scale']


class Logistic(ScipyContinuous):
    """
    Logistic distribution, params are [loc, scale]
    """
    def __init__(self, loc=0, scale=1):
        super().__init__(loc=loc, scale=scale)
        self.scipy_name = stats.logistic


class Pareto(ScipyContinuous):
    """
    Pareto distribution, params are [b, loc, scale]
    """
    def __init__(self, b, loc=0., scale=1.):
        super().__init__(b=b, loc=loc, scale=scale)
        self.scipy_name = stats.pareto
        self.ordered_params = ['b', 'loc', 'scale']


class Rayleigh(ScipyContinuous):
    """
    Logistic distribution, params are [loc, scale]
    """

    def __init__(self, loc=0, scale=1):
        super().__init__(loc=loc, scale=scale)
        self.scipy_name = stats.rayleigh


class Levy(ScipyContinuous):
    """
    Levy distribution, params are [loc, scale]
    """

    def __init__(self, loc=0, scale=1):
        super().__init__(loc=loc, scale=scale)
        self.scipy_name = stats.levy


class Laplace(ScipyContinuous):
    """
    Laplace distribution, params are [loc, scale]
    """

    def __init__(self, loc=0, scale=1):
        super().__init__(loc=loc, scale=scale)
        self.scipy_name = stats.laplace


class Maxwell(ScipyContinuous):
    """
    Maxwell distribution, params are [loc, scale]
    """

    def __init__(self, loc=0, scale=1):
        super().__init__(loc=loc, scale=scale)
        self.scipy_name = stats.maxwell


class TruncNorm(ScipyContinuous):
    """
    Truncated normal distribution, params are [a, b, loc, scale]
    """
    def __init__(self, a, b, loc=0, scale=1.):
        super().__init__(a=a, b=b, loc=loc, scale=scale)
        self.scipy_name = stats.truncnorm
        self.ordered_params = ['a', 'b', 'loc', 'scale']


########################################################################################################################
#        Univariate Discrete Distributions
########################################################################################################################

class DistributionDiscrete1D(Distribution):
    """

    """
    def __init__(self, **kwargs):
        super().__init__()
        self.parameters = kwargs


class ScipyDiscrete(DistributionDiscrete1D):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scipy_name = stats.rv_discrete
        self.ordered_params = ['loc', 'scale']

    def pdf(self, x):
        x = self.check_x_dimension(x)
        return self.scipy_name.pmf(x=x, **self.parameters)

    def log_pdf(self, x):
        x = self.check_x_dimension(x)
        return self.scipy_name.logpmf(x=x, **self.parameters)

    def cdf(self, x):
        x = self.check_x_dimension(x)
        return self.scipy_name.cdf(x=x, **self.parameters)

    def icdf(self, x):
        x = self.check_x_dimension(x)
        return self.scipy_name.ppf(x=x, **self.parameters)

    def rvs(self, nsamples):
        if not isinstance(nsamples, int) and nsamples >= 1:
            raise ValueError('Input nsamples must be an integer strictly greater than 0.')
        tmp_rvs = self.scipy_name.rvs(size=nsamples, **self.parameters)
        return tmp_rvs.reshape((-1, 1))

    def moments(self):
        return self.scipy_name.stats(moments='mvsk', **self.parameters)

    def update_parameters(self, params, fixed_params=None):
        if fixed_params is None:
            fixed_params = {}
        if (len(np.array(params).shape) != 1) or (len(params) + len(fixed_params) != len(self.parameters)):
            raise ValueError('Incorrect number of parameters.')
        cnt_params = 0
        for key in self.ordered_params:
            if key in fixed_params.keys():
                self.parameters[key] = fixed_params[key]
            else:
                self.parameters[key] = params[cnt_params]
                cnt_params += 1


class Binomial(ScipyDiscrete):
    """
    Binomial distribution, params are [n, p]
    """
    def __init__(self, n, p):
        super().__init__(n=n, p=p)
        self.scipy_name = stats.binom
        self.ordered_params = ['n', 'p']


class Poisson(ScipyDiscrete):
    """
    Poisson distribution, params are [mu, loc]
    """
    def __init__(self, mu, loc=0.):
        super().__init__(mu=mu, loc=loc)
        self.scipy_name = stats.poisson
        self.ordered_params = ['mu', 'loc']


########################################################################################################################
#        Multivariate Continuous Distributions
########################################################################################################################

class DistributionND(Distribution):
    """

    Defining multivariate distributions using ``UQpy`` can be easily done via this class. For example, if we want to
    define the ``bivariate Rosenbrock`` distribution model that it is not part of the available ``scipy`` distribution
    models we can do it as:

    >>> from UQpy.Distributions import DistributionND
    >>>
    >>> class rosenbrock(DistributionND):
    >>>     def __init__(self, p=20.):
    >>>         self.parameters = {'p': p}
    >>>     def pdf(self, x):
    >>>         return np.exp(-(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/self.parameters['p'])
    >>>     def log_pdf(self, x):
    >>>          return -(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/self.parameters['p']
    >>> dist = rosenbrock(p=20)
    >>> print(hasattr(dist, 'pdf'))
        True
    >>> print(hasattr(dist, 'rvs'))
        False
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.parameters = kwargs


class MVNormal(DistributionND):

    def __init__(self, mean, cov=1.):
        super().__init__(mean=mean, cov=cov)

    def pdf(self, x):
        pdf_val = stats.multivariate_normal.pdf(x=x, **self.parameters)
        return np.atleast_1d(pdf_val)

    def log_pdf(self, x):
        logpdf_val = stats.multivariate_normal.logpdf(x=x, **self.parameters)
        return np.atleast_1d(logpdf_val)

    def rvs(self, nsamples):
        if not (isinstance(nsamples, int) and nsamples >= 1):
            raise ValueError('Input nsamples must be an integer > 0.')
        return stats.multivariate_normal.rvs(size=nsamples, **self.parameters).reshape((nsamples, -1))

    def mle(self, x):
        mle_mu = np.mean(x, axis=0)
        mle_cov = np.cov(x, rowvar=False, bias=True)
        return [mle_mu, mle_cov]


class Multinomial(DistributionND):
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


class JointInd(DistributionND):
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
        if not (isinstance(marginals, list) and all(isinstance(d, (DistributionContinuous1D, DistributionDiscrete1D))
                                                    for d in marginals)):
            raise ValueError('Input marginals must be a list of Distribution1d objects.')
        self.marginals = marginals

        # If all marginals have a method, the joint has it to
        if all(hasattr(m, 'pdf') or hasattr(m, 'pmf') for m in self.marginals):
            def joint_pdf(dist, x):
                x = dist.check_x_dimension_mv(x)
                # Compute pdf of independent marginals
                pdf_val = np.ones((x.shape[0], ))
                for i in range(len(self.marginals)):
                    if hasattr(self.marginals[i], 'pdf'):
                        pdf_val *= marginals[i].pdf(x[:, i])
                    else:
                        pdf_val *= marginals[i].pmf(x[:, i])
                return pdf_val
            if any(hasattr(m, 'pdf') for m in self.marginals):
                self.pdf = MethodType(joint_pdf, self)
            else:
                self.pmf = MethodType(joint_pdf, self)

        if all(hasattr(m, 'log_pdf') or hasattr(m, 'log_pmf') for m in self.marginals):
            def joint_log_pdf(dist, x):
                x = dist.check_x_dimension_mv(x)
                # Compute pdf of independent marginals
                pdf_val = np.zeros((x.shape[0],))
                for i in range(len(self.marginals)):
                    if hasattr(self.marginals[i], 'log_pdf'):
                        pdf_val += marginals[i].log_pdf(x[:, i])
                    else:
                        pdf_val += marginals[i].log_pmf(x[:, i])
                return pdf_val
            if any(hasattr(m, 'log_pdf') for m in self.marginals):
                self.log_pdf = MethodType(joint_log_pdf, self)
            else:
                self.log_pmf = MethodType(joint_log_pdf, self)

        if all(hasattr(m, 'cdf') for m in self.marginals):
            def joint_cdf(dist, x):
                x = dist.check_x_dimension_mv(x)
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

        if all(hasattr(m, 'update_parameters') for m in self.marginals):
            def joint_update_parameters(dist, params, fixed_params=None):
                if fixed_params is None:
                    fixed_params = [{}] * len(self.marginals)
                if not isinstance(fixed_params, (list, tuple)) or len(fixed_params) != len(self.marginals):
                    raise ValueError
                cnt_params = 0
                for m, fixed_p in zip(dist.marginals, fixed_params):
                    n_free = len(m.parameters) - len(fixed_p)
                    m.update_parameters(params=np.atleast_1d(params[cnt_params:cnt_params+n_free]),
                                        fixed_params=fixed_p)
                    cnt_params += n_free
            self.update_parameters = MethodType(joint_update_parameters, self)


class JointCopula(DistributionND):
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
        #TODO: Consider copulas for non-continuous distributions with evaluate_pmf?
        self.marginals = marginals
        if not (isinstance(self.marginals, list)
                and all(isinstance(d, (DistributionContinuous1D, DistributionDiscrete1D)) for d in self.marginals)):
            raise ValueError('Input marginals must be a list of 1d continuous Distribution objects.')

        # Check the copula. Also, all the marginals should have a cdf method
        self.copula = copula
        if not isinstance(self.copula, Copula):
            raise ValueError('The input copula should be a Copula object.')
        if not all(hasattr(m, 'cdf') for m in self.marginals):
            raise ValueError('All the marginals should have a cdf method in order to define a joint with copula.')
        self.copula.check_marginals(marginals=self.marginals)

        # Check if methods should exist, if yes define them bound them to the object
        if hasattr(self.copula, 'evaluate_cdf'):
            def joint_cdf(dist, x):
                x = dist.check_x_dimension_mv(x)
                # Compute cdf of independent marginals
                unif = np.array([m.cdf(x[:, i]) for i, m in enumerate(dist.marginals)]).T
                # Compute copula
                cdf_val = dist.copula.evaluate_cdf(unif=unif)
                return cdf_val
            self.cdf = MethodType(joint_cdf, self)

        if all(hasattr(m, 'pdf') for m in self.marginals) and hasattr(self.copula, 'evaluate_pdf'):
            def joint_pdf(dist, x):
                x = dist.check_x_dimension_mv(x)
                # Compute pdf of independent marginals
                pdf_val = np.prod(np.array([m.pdf(x[:, i]) for i, m in enumerate(dist.marginals)]), axis=0)
                # Add copula term
                unif = np.array([m.cdf(x[:, i]) for i, m in enumerate(dist.marginals)]).T
                c_ = dist.copula.evaluate_pdf(unif=unif)
                return c_ * pdf_val
            self.pdf = MethodType(joint_pdf, self)

        if all(hasattr(m, 'log_pdf') for m in self.marginals) and hasattr(self.copula, 'evaluate_pdf'):
            def joint_log_pdf(dist, x):
                x = dist.check_x_dimension_mv(x)
                # Compute pdf of independent marginals
                logpdf_val = np.sum(np.array([m.log_pdf(x[:, i]) for i, m in enumerate(dist.marginals)]), axis=0)
                # Add copula term
                unif = np.array([m.cdf(x[:, i]) for i, m in enumerate(dist.marginals)]).T
                c_ = dist.copula.evaluate_pdf(unif=unif)
                return np.log(c_) + logpdf_val
            self.log_pdf = MethodType(joint_log_pdf, self)

        if all(hasattr(m, 'update_parameters') for m in self.marginals + [self.copula, ]):
            def joint_update_parameters(dist, params, fixed_params=None):
                if fixed_params is None:
                    fixed_params = [{}] * (len(self.marginals) + 1)
                if not isinstance(fixed_params, (list, tuple)) or len(fixed_params) != len(self.marginals) + 1:
                    raise ValueError
                cnt_params = 0
                for m, fixed_p in zip(dist.marginals + [dist.copula, ], fixed_params):
                    n_free = len(m.parameters) - len(fixed_p)
                    m.update_parameters(params=np.atleast_1d(params[cnt_params:cnt_params+n_free]),
                                        fixed_params=fixed_p)
                    cnt_params += n_free
            self.update_parameters = MethodType(joint_update_parameters, self)


########################################################################################################################
#        Copulas
########################################################################################################################

class Copula:
    """
    Parent class to all copulas
    """

    def __init__(self):
        pass

    def __evaluate_cdf(self, unif):
        """
        Compute the copula cdf ``C(u1, u2, ..., ud)`` for a d-variate uniform distribution.

        For a generic multivariate distribution with marginals ``F1, ...Fd``, the joint cdf is computed as
        ``F(x_1, ..., x_d) = C(F_1(x_1), ..., F_d(x_d))``. Thus one must first evaluate the marginals cdf, then
        evaluate the copula cdf.

        **Input:**

                unif (ndarray):
                            Points (uniformly distributed) at which to evaluate the copula *cdf*. ``unif.shape`` must
                            be  ``(npoints, dimension)`` .

        **Output/Returns:**

                (ndarray):
                        The values of the copula's cdf. Is an array of shape ``(npoints, )``

        """
        pass

    def __evaluate_pdf(self, unif):
        """
        Compute the copula pdf term ``C_`` for a d-variate uniform distribution.

        For a generic multivariate distribution with marginals pdfs f1, ..., fd, the joint pdf is computed as
        ``f(x1, ..., xd) = C_ * f1(x1) * ... * fd(xd)``. Thus one must first evaluate the marginals cdf and copula term
        ``C_``, them multiply it by the marginal pdfs.

        **Input:**

                unif (ndarray):
                            Points (uniformly distributed) at which to evaluate the copula *pdf* term ``C_``.
                             ``unif.shape`` must be  ``(npoints, dimension)`` .

        **Output/Returns:**

                (ndarray):
                        The values of the copula's pdf. Is an array of shape ``(npoints, )``
        """
        pass


class Gumbel(Copula):
    """
    Gumbel copula
    """
    def __init__(self, **kwargs):
        self.parameters = kwargs
        theta = self.parameters['theta']
        # Check the input copula_params
        if theta is not None and ((not isinstance(theta, (float, int))) or (theta < 1)):
            raise ValueError('Input theta should be a float in [1, +oo).')
        super().__init__()

    def evaluate_cdf(self, unif):
        if unif.shape[1] > 2:
            raise ValueError('Maximum dimension for the Gumbel Copula is 2.')
        if self.parameters['theta'] == 1:
            return np.prod(unif, axis=1)

        u = unif[:, 0]
        v = unif[:, 1]
        theta = self.parameters['theta']
        cdf_val = np.exp(-((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta))

        return cdf_val

    def evaluate_pdf(self, unif):
        if unif.shape[1] > 2:
            raise ValueError('Maximum dimension for the Gumbel Copula is 2.')
        if self.parameters['theta'] == 1:
            return np.ones(unif.shape[0])

        u = unif[:, 0]
        v = unif[:, 1]
        theta = self.parameters['theta']
        c = np.exp(-((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta))

        pdf_val = c * 1 / u * 1 / v * ((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (-2 + 2 / theta) \
             * (np.log(u) * np.log(v)) ** (theta - 1) * \
             (1 + (theta - 1) * ((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (-1 / theta))
        return pdf_val

    def check_marginals(self, marginals):
        if len(marginals) != 2:
            raise ValueError('Maximum dimension for the Gumbel Copula is 2.')
        if not all(isinstance(m, DistributionContinuous1D) for m in marginals):
            raise ValueError('Marginals should be 1d continuous distributions.')

    def update_parameters(self, params, **fixed_params):
        if 'theta' in fixed_params.keys():
            self.parameters['theta'] = fixed_params['theta']
        else:
            self.parameters['theta'] = params[0]


class Clayton(Copula):
    """
    Clayton copula
    """
    def __init__(self, **kwargs):
        self.parameters = kwargs
        theta = self.parameters['theta']
        # Check the input copula_params
        if theta is not None and ((not isinstance(theta, (float, int))) or (theta < -1 or theta == 0.)):
            raise ValueError('Input theta should be a float in [-1, +oo)\{0}.')
        super().__init__()

    def evaluate_cdf(self, unif):
        if unif.shape[1] > 2:
            raise ValueError('Maximum dimension for the Clayton Copula is 2.')
        if self.parameters['theta'] == 1:
            return np.prod(unif, axis=1)

        u = unif[:, 0]
        v = unif[:, 1]
        theta = self.parameters['theta']
        cdf_val = (np.maximum(u ** (-theta) + v ** (-theta) - 1., 0.)) ** (-1. / theta)
        return cdf_val

    def check_marginals(self, marginals):
        if len(marginals) != 2:
            raise ValueError('Maximum dimension for the Clayton Copula is 2.')
        if not all(isinstance(m, DistributionContinuous1D) for m in marginals):
            raise ValueError('Marginals should be 1d continuous distributions.')

    def update_parameters(self, params, **fixed_params):
        if 'theta' in fixed_params.keys():
            self.parameters['theta'] = fixed_params['theta']
        else:
            self.parameters['theta'] = params[0]


class Frank(Copula):
    """
    Frank copula
    """
    def __init__(self, **kwargs):
        self.parameters = kwargs
        theta = self.parameters['theta']
        # Check the input copula_params
        if theta is not None and ((not isinstance(theta, (float, int))) or (theta == 0.)):
            raise ValueError('Input theta should be a float in R\{0}.')
        super().__init__()

    def evaluate_cdf(self, unif):
        if unif.shape[1] > 2:
            raise ValueError('Maximum dimension for the Clayton Copula is 2.')
        if self.parameters['theta'] == 1:
            return np.prod(unif, axis=1)

        u = unif[:, 0]
        v = unif[:, 1]
        theta = self.parameters['theta']
        tmp_ratio = (np.exp(-theta * u) - 1.) * (np.exp(-theta * v) - 1.) / (np.exp(-theta) - 1.)
        cdf_val = -1. / theta * np.log(1. + tmp_ratio)
        return cdf_val

    def check_marginals(self, marginals):
        if len(marginals) != 2:
            raise ValueError('Maximum dimension for the Frank Copula is 2.')
        if not all(isinstance(m, DistributionContinuous1D) for m in marginals):
            raise ValueError('Marginals should be 1d continuous distributions.')

    def update_parameters(self, params, **fixed_params):
        if 'theta' in fixed_params.keys():
            self.parameters['theta'] = fixed_params['theta']
        else:
            self.parameters['theta'] = params[0]
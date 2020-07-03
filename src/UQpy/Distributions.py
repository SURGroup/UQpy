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
This module contains functionality for all probability distributions supported in ``UQpy``.

The ``Distributions`` module is  used  to  define  probability  distribution  objects.   These  objects  possess various
methods  that  allow the user  to:  compute  the  probability  density/mass  function ``pdf/pmf``, the cumulative
distribution  function ``cdf``, the logarithm of the pdf/pmf ``log_pdf/log_pmf``, return the moments ``moments``, draw
independent samples ``rvs`` and compute the maximum likelihood estimate of the parameters from data ``mle``.

The module contains the following parent classes - probability distributions are defined via sub-classing those parent
classes:

- ``Distribution``: Parent class to all distributions.
- ``DistributionContinuous1D``: Parent class to 1-dimensional continuous probability distributions.
- ``DistributionDiscrete1D``: Parent class to 1-dimensional discrete probability distributions.
- ``DistributionND``: Parent class to multivariate probability distributions.
- ``Copula``: Parent class to copula to model dependency between marginals.


"""

import importlib
import os
from types import MethodType

import numpy as np
import scipy.stats as stats


########################################################################################################################
#        Define the probability distribution of the random parameters
########################################################################################################################


class Distribution:
    """
    A parent class to all ``Distribution`` classes.

    All distributions possess a number of methods to perform basic probabilistic operations. For most of the predefined
    distributions in ``UQpy`` these methods are inherited from the ``scipy.stats`` package. These include standard
    operations such as computing probability density/mass functions, cumulative distribution functions and their
    inverse, drawing random samples, computing moments and parameter fitting. However, for user-defined distributions,
    any desired method can be constructed into the child class structure.

    For bookkeeping purposes, all ``Distribution`` objects possesses ``get_params`` and ``update_params`` methods. These
    are described in more detail below.

    Any ``Distribution`` further inherits from one of the following classes:

    - ``DistributionContinuous1D``: Parent class to 1-dimensional continuous probability distributions.
    - ``DistributionDiscrete1D``: Parent class to 1-dimensional discrete probability distributions.
    - ``DistributionND``: Parent class to multivariate probability distributions.


    **Attributes:**

    * **order_params** (`list`):
        Ordered list of parameter names, useful when parameter values are stored in vectors and must be passed to the
        ``update_params`` method.

    * **params** (`dict`):
        Parameters of the distribution. Note: this attribute is not defined for certain ``Distribution`` objects such as
        those of type ``JointInd`` or ``JointCopula``. The user is advised to use the ``get_params`` method to access
        the parameters.

    **Methods:**

    **cdf** *(x)*
        Evaluate the cumulative distribution function.

        **Input:**

        * **x** (`ndarray`):
            Point(s) at which to evaluate the `cdf`, must be of shape `(npoints, dimension)`.

        **Output/Returns:**

        * (`ndarray`):
            Evaluated cdf values, `ndarray` of shape `(npoints,)`.

    **pdf** *(x)*
        Evaluate the probability density function of a continuous or multivariate mixed continuous-discrete
        distribution.

        **Input:**

        * **x** (`ndarray`):
            Point(s) at which to evaluate the `pdf`, must be of shape `(npoints, dimension)`.

        **Output/Returns:**

        * (`ndarray`):
            Evaluated pdf values, `ndarray` of shape `(npoints,)`.

    **pmf** *(x)*
        Evaluate the probability mass function of a discrete distribution.

        **Input:**

        * **x** (`ndarray`):
            Point(s) at which to evaluate the `pmf`, must be of shape `(npoints, dimension)`.

        **Output/Returns:**

        * (`ndarray`):
            Evaluated pmf values, `ndarray` of shape `(npoints,)`.

    **log_pdf** *(x)*
        Evaluate the logarithm of the probability density function of a continuous or multivariate mixed
        continuous-discrete distribution.

        **Input:**

        * **x** (`ndarray`):
            Point(s) at which to evaluate the `log_pdf`, must be of shape `(npoints, dimension)`.

        **Output/Returns:**

        * (`ndarray`):
            Evaluated log-pdf values, `ndarray` of shape `(npoints,)`.

    **log_pmf** *(x)*
        Evaluate the logarithm of the probability mass function of a discrete distribution.

        **Input:**

        * **x** (`ndarray`):
            Point(s) at which to evaluate the `log_pmf`, must be of shape `(npoints, dimension)`.

        **Output/Returns:**

        * (`ndarray`):
            Evaluated log-pmf values, `ndarray` of shape `(npoints,)`.

    **icdf** *(x)*
        Evaluate the inverse cumulative distribution function for univariate distributions.

        **Input:**

        * **x** (`ndarray`):
            Point(s) at which to evaluate the `icdf`, must be of shape `(npoints, dimension)`.

        **Output/Returns:**

        * (`ndarray`):
            Evaluated icdf values, `ndarray` of shape `(npoints,)`.

    **rvs** *(nsamples=1, random_state=None)*
        Sample independent identically distributed (iid) realizations.

        **Inputs:**

        * **nsamples** (`int`):
            Number of iid samples to be drawn. Default is 1.

        * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
            Random seed used to initialize the pseudo-random number generator. Default is None.

            If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
            object itself can be passed directly.

        **Output/Returns:**

        * (`ndarray`):
            Generated iid samples, `ndarray` of shape `(npoints, dimension)`.

    **moments** *(moments2return='mvsk')*
        Computes the mean 'number_of_variables', variance/covariance ('v'), skewness ('s') and/or kurtosis ('k') of the distribution.

        For a univariate distribution, mean, variance, skewness and kurtosis are returned. For a multivariate
        distribution, the mean vector, covariance and vectors of marginal skewness and marginal kurtosis are returned.

        **Inputs:**

        * **moments2return** (`str`):
            Indicates which moments are to be returned (mean, variance, skewness and/or kurtosis). Default is 'mvsk'.

        **Output/Returns:**

        * (`tuple`):
            ``mean``: mean, ``var``:  variance/covariance, ``skew``: skewness, ``kurt``: kurtosis.

    **fit** *(data)*
        Compute the maximum-likelihood parameters from iid data.

        Computes the mle analytically if possible. For univariate continuous distributions, it leverages the fit
        method of the scipy.stats package.

        **Input:**

        * **data** (`ndarray`):
            Data array, must be of shape `(npoints, dimension)`.

        **Output/Returns:**

        * (`dict`):
            Maximum-likelihood parameter estimates.

    """
    def __init__(self, order_params=None, **kwargs):
        self.params = kwargs
        self.order_params = order_params
        if self.order_params is None:
            self.order_params = tuple(kwargs.keys())
        if len(self.order_params) != len(self.params):
            raise ValueError('Inconsistent dimensions between order_params tuple and params dictionary.')

    def update_params(self, **kwargs):
        """
        Update the parameters of a ``Distributions`` object.

        To update the parameters of a ``JointInd`` or a ``JointCopula`` distribution, each parameter is assigned a
        unique string identifier as `key_index` - where `key` is the parameter name and `index` the index of the
        marginal (e.g., location parameter of the 2nd marginal is identified as `loc_1`).

        **Input:**

        * keyword arguments:
            Parameters to be updated, designated by their respective keywords.

        """
        for key in kwargs.keys():
            if key not in self.get_params().keys():
                raise ValueError('Wrong parameter name.')
            self.params[key] = kwargs[key]

    def get_params(self):
        """
        Return the parameters of a ``Distributions`` object.

        To update the parameters of a ``JointInd`` or a ``JointCopula`` distribution, each parameter is assigned a
        unique string identifier as `key_index` - where `key` is the parameter name and `index` the index of the
        marginal (e.g., location parameter of the 2nd marginal is identified as `loc_1`).

        **Output/Returns:**

        * (`dict`):
            Parameters of the distribution.

        """
        return self.params


class DistributionContinuous1D(Distribution):
    """
    Parent class for univariate continuous probability distributions.


    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _check_x_dimension(x):
        """
        Check the dimension of input x - must be an ndarray of shape (npoints,) or (npoints, 1)
        """
        x = np.atleast_1d(x)
        if len(x.shape) > 2 or (len(x.shape) == 2 and x.shape[1] != 1):
            raise ValueError('Wrong dimension in x.')
        return x.reshape((-1,))

    def _construct_from_scipy(self, scipy_name=stats.rv_continuous):
        self.cdf = lambda x: scipy_name.cdf(x=self._check_x_dimension(x), **self.params)
        self.pdf = lambda x: scipy_name.pdf(x=self._check_x_dimension(x), **self.params)
        self.log_pdf = lambda x: scipy_name.logpdf(x=self._check_x_dimension(x), **self.params)
        self.icdf = lambda x: scipy_name.ppf(q=self._check_x_dimension(x), **self.params)
        self.moments = lambda moments2return='mvsk': scipy_name.stats(moments=moments2return, **self.params)
        self.rvs = lambda nsamples=1, random_state=None: scipy_name.rvs(
            size=nsamples, random_state=random_state, **self.params).reshape((nsamples, 1))

        def tmp_fit(dist, data):
            data = self._check_x_dimension(data)
            fixed_params = {}
            for key, value in dist.params.items():
                if value is not None:
                    fixed_params['f' + key] = value
            params_fitted = scipy_name.fit(data=data, **fixed_params)
            return dict(zip(dist.order_params, params_fitted))
        self.fit = lambda data: tmp_fit(self, data)


########################################################################################################################
#        Univariate Continuous Distributions
########################################################################################################################


class Beta(DistributionContinuous1D):
    """
    Beta distribution having probability density function

    .. math:: f(x|a,b) = \dfrac{\Gamma(a+b)x^{a-1}(1-x)^{b-1}}{\Gamma(a)\Gamma(b)}

    for :math:`0\le x\ge 0`, :math:`a>0, b>0`. Here :math:`\Gamma(a)` refers to the Gamma function.

    In this standard form `(loc=0, scale=1)`, the distribution is defined over the interval (0, 1). Use `loc` and
    `scale` to shift the distribution to interval `(loc, loc + scale)`. Specifically, this is equivalent to computing
    :math:`f(y|a,b)` where :math:`y=(x-loc)/scale`.

    **Inputs:**

    * **a** (`float`):
        first shape parameter
    * **b** (float):
        second shape parameter
    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``Beta``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``
    """
    def __init__(self, a, b, loc=0., scale=1.):
        super().__init__(a=a, b=b, loc=loc, scale=scale, order_params=('a', 'b', 'loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.beta)


class Cauchy(DistributionContinuous1D):
    """
    Cauchy distribution having probability density function

    .. math:: f(x) = \dfrac{1}{\pi(1+x^2)}

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    **Inputs:**

    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``Cauchy``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``
    """
    def __init__(self, loc=0., scale=1.):
        super().__init__(loc=loc, scale=scale, order_params=('loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.cauchy)


class ChiSquare(DistributionContinuous1D):
    """
    Chi-square distribution having probability density:

    .. math:: f(x|k) = \dfrac{1}{2^{k/2}\Gamma(k/2)}x^{k/2-1}\exp{(-x/2)}

    for :math:`x\ge 0`, :math:`k>0`. Here :math:`\Gamma(\cdot)` refers to the Gamma function.

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y|k)` where :math:`y=(x-loc)/scale`.

    **Inputs:**

    * **df** (`float`):
        shape parameter (degrees of freedom) (given by `k` in the equation above)
    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``ChiSquare``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, df, loc=0., scale=1):
        super().__init__(df=df, loc=loc, scale=scale, order_params=('df', 'loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.chi2)


class Exponential(DistributionContinuous1D):
    """
    Exponential distribution having probability density function:

    .. math:: f(x) = \exp(-x)

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    A common parameterization for Exponential is in terms of the rate parameter :math:`\lambda`, which corresponds to
    using :math:`scale = 1 / \lambda`.

    **Inputs:**

    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``Exponential``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, loc=0., scale=1.):
        super().__init__(loc=loc, scale=scale, order_params=('loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.expon)


class Gamma(DistributionContinuous1D):
    """
    Gamma distribution having probability density function:

    .. math:: f(x|a) = \dfrac{x^{a-1}\exp(-x)}{\Gamma(a)}

    for :math:`x\ge 0`, :math:`a>0`. Here :math:`\Gamma(a)` refers to the Gamma function.

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    **Inputs:**

    * **a** (`float`):
        shape parameter
    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``Gamma``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, a, loc=0., scale=1.):
        super().__init__(a=a, loc=loc, scale=scale, order_params=('a', 'loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.gamma)


class GenExtreme(DistributionContinuous1D):
    """
    Generalized Extreme Value distribution having probability density function:

    .. math:: `f(x|c) = \exp(-(1-cx)^{1/c})(1-cx)^{1/c-1}`

    for :math:`x\le 1/c, c>0`.

    For `c=0`

    .. math:: f(x) = \exp(\exp(-x))\exp(-x)

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    **Inputs:**

    * **c** (`float`):
        shape parameter
    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``GenExtreme``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, c, loc=0., scale=1.):
        super().__init__(c=c, loc=loc, scale=scale, order_params=('c', 'loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.genextreme)


class InvGauss(DistributionContinuous1D):
    """
    Inverse Gaussian distribution having probability density function

    .. math:: f(x|\mu) = \dfrac{1}{2\pi x^3}\exp{(-\dfrac{(x\\mu)^2}{2x\mu^2})}

    for :math:`x>0`. ``cdf`` method returns `NaN` for :math:`\mu<0.0028`.

    **Inputs:**

    * **mu** (`float`):
        shape parameter, :math:`\mu`
    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    The following methods are available for ``InvGauss``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, mu, loc=0., scale=1.):
        super().__init__(mu=mu, loc=loc, scale=scale, order_params=('mu', 'loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.invgauss)


class Laplace(DistributionContinuous1D):
    """
    Laplace distribution having probability density function

    .. math:: f(x) = \dfrac{1}{2}\exp{-|x|}

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    **Inputs:**

    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``Laplace``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, loc=0, scale=1):
        super().__init__(loc=loc, scale=scale, order_params=('loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.laplace)


class Levy(DistributionContinuous1D):
    """
    Levy distribution having probability density function

    .. math:: f(x) = \dfrac{1}{\sqrt{2\pi x^3}}\exp(-\dfrac{1}{2x})

    for :math:`x\ge 0`.

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    **Inputs:**

    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``Levy``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, loc=0, scale=1):
        super().__init__(loc=loc, scale=scale, order_params=('loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.levy)


class Logistic(DistributionContinuous1D):
    """
    Logistic distribution having probability density function

    .. math:: f(x) = \dfrac{\exp(-x)}{(1+\exp(-x))^2}

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    **Inputs:**

    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``Logistic``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, loc=0, scale=1):
        super().__init__(loc=loc, scale=scale, order_params=('loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.logistic)


class Lognormal(DistributionContinuous1D):
    """
    Lognormal distribution having probability density function

    .. math:: f(x|s) = \dfrac{1}{sx\sqrt{2\pi}}\exp(-\dfrac{\log^2(x)}{2s^2})

    for :math:`x>0, s>0`.

    A common parametrization for a lognormal random variable Y is in terms of the mean, mu, and standard deviation,
    sigma, of the gaussian random variable X such that exp(X) = Y. This parametrization corresponds to setting
    s = sigma and scale = exp(mu).

    **Inputs:**

    * **s** (`float`):
        shape parameter
    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``Lognormal``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, s, loc=0., scale=1.):
        super().__init__(s=s, loc=loc, scale=scale, order_params=('s', 'loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.lognorm)


class Maxwell(DistributionContinuous1D):
    """
    Maxwell-Boltzmann distribution having probability density function

    .. math:: f(x) = \sqrt{2/\pi}x^2\exp(-x^2/2)

    for :math:`x\ge0`.

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    **Inputs:**

    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``Maxwell``:
    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, loc=0, scale=1):
        super().__init__(loc=loc, scale=scale, order_params=('loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.maxwell)


class Normal(DistributionContinuous1D):
    """
    Normal distribution having probability density function

    .. math:: f(x) = \dfrac{\exp(-x^2/2)}{\sqrt{2\pi}}

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    **Inputs:**

    * **loc** (`float`):
        mean
    * **scale** (`float`):
        standard deviation

    The following methods are available for ``Normal``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, loc=0., scale=1.):
        super().__init__(loc=loc, scale=scale, order_params=('loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.norm)

    def fit(self, x):
        x = self._check_x_dimension(x)
        mle_loc, mle_scale = self.params['loc'], self.params['scale']
        if mle_loc is None:
            mle_loc = np.mean(x)
        if mle_scale is None:
            mle_scale = np.sqrt(np.mean((x - mle_loc) ** 2))
        return {'loc': mle_loc, 'scale': mle_scale}


class Pareto(DistributionContinuous1D):
    """
    Pareto distribution having probability density function

    .. math:: f(x|b) = \dfrac{b}{x^{b+1}}

    for :math:`x\ge 1, b>0`.

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    **Inputs:**

    * **b** (`float`):
        shape parameter
    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``Pareto``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, b, loc=0., scale=1.):
        super().__init__(b=b, loc=loc, scale=scale, order_params=('b', 'loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.pareto)


class Rayleigh(DistributionContinuous1D):
    """
    Rayleigh distribution having probability density function

    .. math:: f(x) = x\exp(-x^2/2)

    for :math:`x\ge 0`.

    In this standard form `(loc=0, scale=1)`. Use `loc` and `scale` to shift and scale the distribution. Specifically,
    this is equivalent to computing :math:`f(y)` where :math:`y=(x-loc)/scale`.

    **Inputs:**

    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``Rayleigh``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, loc=0, scale=1):
        super().__init__(loc=loc, scale=scale, order_params=('loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.rayleigh)


class TruncNorm(DistributionContinuous1D):
    """
    Truncated normal distribution

    The standard form of this distribution (i.e, loc=0., scale=1) is a standard normal truncated to the range [a, b].
    Note that a and b are defined over the domain of the standard normal.

    **Inputs:**

    * **a** (`float`):
        shape parameter
    * **b** (`float`):
        shape parameter
    * **loc** (`float`):
        location parameter
    * **scale** (`float`):
        scale parameter

    The following methods are available for ``TruncNorm``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, a, b, loc=0, scale=1.):
        super().__init__(a=a, b=b, loc=loc, scale=scale, order_params=('a', 'b', 'loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.truncnorm)


class Uniform(DistributionContinuous1D):
    """
    Uniform distribution having probability density function

    .. math:: f(x|a, b) = \dfrac{1}{b-a}

    where :math:`a=loc` and :math:`b=loc+scale`

    **Inputs:**

    * **loc** (`float`):
        lower bound
    * **scale** (`float`):
        range

    The following methods are available for ``Uniform``:

    * ``cdf``, ``pdf``, ``log_pdf``, ``icdf``, ``rvs``, ``moments``, ``fit``.
    """
    def __init__(self, loc=0., scale=1.):
        super().__init__(loc=loc, scale=scale, order_params=('loc', 'scale'))
        self._construct_from_scipy(scipy_name=stats.uniform)


########################################################################################################################
#        Univariate Discrete Distributions
########################################################################################################################

class DistributionDiscrete1D(Distribution):
    """
    Parent class for univariate discrete distributions.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _check_x_dimension(x):
        """
        Check the dimension of input x - must be an ndarray of shape (npoints,) or (npoints, 1)
        """
        x = np.atleast_1d(x)
        if len(x.shape) > 2 or (len(x.shape) == 2 and x.shape[1] != 1):
            raise ValueError('Wrong dimension in x.')
        return x.reshape((-1, ))

    def _construct_from_scipy(self, scipy_name=stats.rv_discrete):
        self.cdf = lambda x: scipy_name.cdf(x=self._check_x_dimension(x), **self.params)
        self.pmf = lambda x: scipy_name.pmf(x=self._check_x_dimension(x), **self.params)
        self.log_pmf = lambda x: scipy_name.logpmf(x=self._check_x_dimension(x), **self.params)
        self.icdf = lambda x: scipy_name.ppf(q=self._check_x_dimension(x), **self.params)
        self.moments = lambda moments2return='mvsk': scipy_name.stats(moments=moments2return, **self.params)
        self.rvs = lambda nsamples=1, random_state=None: scipy_name.rvs(
            size=nsamples, random_state=random_state, **self.params).reshape((nsamples, 1))


class Binomial(DistributionDiscrete1D):
    """
    Binomial distribution having probability mass function:

    .. math:: f(x) = {number_of_dimensions \choose x} p^x(1-p)^{number_of_dimensions-x}

    for :math:`x\inumber_of_dimensions\{0, 1, 2, ..., number_of_dimensions\}`.

    In this standard form `(loc=0)`. Use `loc` to shift the distribution. Specifically, this is equivalent to computing
    :math:`f(y)` where :math:`y=x-loc`.

    **Inputs:**

    * **number_of_dimensions** (`int`):
        number of trials, integer >= 0
    * **p** (`float`):
        success probability for each trial, real number in [0, 1]
    * **loc** (`float`):
        location parameter

    The following methods are available for ``Binomial``:

    * ``cdf``, ``pmf``, ``log_pmf``, ``icdf``, ``rvs, moments``.
    """
    def __init__(self, n, p, loc=0.):
        super().__init__(n=n, p=p, loc=loc, order_params=('number_of_dimensions', 'p', 'loc'))
        self._construct_from_scipy(scipy_name=stats.binom)


class Poisson(DistributionDiscrete1D):
    """
    Poisson distribution having probability mass function:

    .. math:: f(x) = \exp{(-\mu)}\dfrac{\mu^k}{k!}

    for :math:`x\ge 0`.

    In this standard form `(loc=0)`. Use `loc` to shift the distribution. Specifically, this is equivalent to computing
    :math:`f(y)` where :math:`y=x-loc`.

    **Inputs:**

    * **mu** (`float`):
        shape parameter
    * **loc** (`float`):
        location parameter

    The following methods are available for ``Poisson``:

    * ``cdf``, ``pmf``, ``log_pmf``, ``icdf``, ``rvs``, ``moments``.
    """
    def __init__(self, mu, loc=0.):
        super().__init__(mu=mu, loc=loc, order_params=('mu', 'loc'))
        self._construct_from_scipy(scipy_name=stats.poisson)


########################################################################################################################
#        Multivariate Continuous Distributions
########################################################################################################################

class DistributionND(Distribution):
    """
    Parent class for multivariate probability distributions.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _check_x_dimension(x, d=None):
        """
        Check the dimension of input x - must be an ndarray of shape (npoints, d)
        """
        x = np.array(x)
        if len(x.shape) != 2:
            raise ValueError('Wrong dimension in x.')
        if (d is not None) and (x.shape[1] != d):
            raise ValueError('Wrong dimension in x.')
        return x


class MVNormal(DistributionND):
    """
    Multivariate normal distribution having probability density function

    .. math:: f(x) = \dfrac{1}{\sqrt{(2\pi)^k\det\Sigma}}\exp{-\dfrac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}

    where :math:`\mu` is the mean vector, :math:`\Sigma` is the covariance matrix, and :math:`k` is the dimension of
    `x`.

    **Inputs:**

    * **mean** (`ndarray`):
        mean vector, `ndarray` of shape `(dimension, )`
    * **cov** (`float` or `ndarray`):
        covariance, `float` or `ndarray` of shape `(dimension, )` or `(dimension, dimension)`. Default is 1.

    The following methods are available for ``MVNormal``:

    * ``pdf``, ``log_pdf``, ``rvs``, ``fit``, ``moments``.
    """
    def __init__(self, mean, cov=1.):
        if len(np.array(mean).shape) != 1:
            raise ValueError('Input mean must be a 1D array.')
        if isinstance(cov, (int, float)):
            pass
        else:
            if not (len(np.array(cov).shape) in [1, 2] and all(sh == len(mean) for sh in np.array(cov).shape)):
                raise ValueError('Input cov must be a float or ndarray of appropriate dimensions.')
        super().__init__(mean=mean, cov=cov, order_params=['mean', 'cov'])

    def pdf(self, x):
        pdf_val = stats.multivariate_normal.pdf(x=x, **self.params)
        return np.atleast_1d(pdf_val)

    def log_pdf(self, x):
        logpdf_val = stats.multivariate_normal.logpdf(x=x, **self.params)
        return np.atleast_1d(logpdf_val)

    def rvs(self, nsamples=1, random_state=None):
        if not (isinstance(nsamples, int) and nsamples >= 1):
            raise ValueError('Input nsamples must be an integer > 0.')
        return stats.multivariate_normal.rvs(
            size=nsamples, random_state=random_state, **self.params).reshape((nsamples, -1))

    def fit(self, x):
        mle_mu, mle_cov = self.params['mean'], self.params['cov']
        if mle_mu is None:
            mle_mu = np.mean(x, axis=0)
        if mle_cov is None:
            tmp_x = x - np.tile(mle_mu.reshape(1, -1), [x.shape[0], 1])
            mle_cov = np.matmul(tmp_x, tmp_x.T) / x.shape[0]
        return {'mean': mle_mu, 'cov': mle_cov}

    def moments(self, moments2return='mv'):
        if moments2return == 'number_of_variables':
            return self.get_params()['mean']
        elif moments2return == 'v':
            return self.get_params()['cov']
        elif moments2return == 'mv':
            return self.get_params()['mean'], self.get_params()['cov']
        else:
            raise ValueError('UQpy: moments2return must be "number_of_variables", "v" or "mv".')


class Multinomial(DistributionND):
    """
    Multinomial distribution having probability mass function

    .. math:: f(x) = \dfrac{number_of_dimensions!}{x_1!\dots x_k!}p_1^{x_1}\dots p_k^{x_k}

    for :math:`x=\{x_1,\dots,x_k\}` where each :math:`x_i` is a non-negative integer and :math:`\sum_i x_i = number_of_dimensions`.

    **Inputs:**

    * **number_of_dimensions** (`int`):
        number of trials
    * **p** (`array_like`):
        probability of a trial falling into each category; should sum to 1

    The following methods are available for ``Multinomial``:

    * ``pmf``, ``log_pmf``, ``rvs``, ``moments``.
    """
    def __init__(self, n, p):
        super().__init__(n=n, p=p)

    def pmf(self, x):
        pdf_val = stats.multinomial.pmf(x=x, **self.params)
        return np.atleast_1d(pdf_val)

    def log_pmf(self, x):
        logpdf_val = stats.multinomial.logpmf(x=x, **self.params)
        return np.atleast_1d(logpdf_val)

    def rvs(self, nsamples=1, random_state=None):
        if not (isinstance(nsamples, int) and nsamples >= 1):
            raise ValueError('Input nsamples must be an integer > 0.')
        return stats.multinomial.rvs(
            size=nsamples, random_state=random_state, **self.params).reshape((nsamples, -1))

    def moments(self, moments2return='mv'):
        if moments2return == 'number_of_variables':
            mean = self.get_params()['number_of_dimensions'] * np.array(self.get_params()['p'])
            return mean
        elif moments2return == 'v':
            n, p = self.get_params()['number_of_dimensions'], np.array(self.get_params()['p'])
            d = len(p)
            cov = - n * np.tile(p[np.newaxis, :], [d, 1]) * np.tile(p[:, np.newaxis], [1, d])
            np.fill_diagonal(cov, n * p * (1. - p))
            return cov
        elif moments2return == 'mv':
            n, p = self.get_params()['number_of_dimensions'], np.array(self.get_params()['p'])
            d = len(p)
            cov = - n * np.tile(p[np.newaxis, :], [d, 1]) * np.tile(p[:, np.newaxis], [1, d])
            np.fill_diagonal(cov, n * p * (1. - p))
            mean = n * p
            return mean, cov
        else:
            raise ValueError('UQpy: moments2return must be "number_of_variables", "v" or "mv".')


class JointInd(DistributionND):
    """
    Define a joint distribution from its independent marginals. ``JointInd`` is a child class of ``DistributionND``.

    **Inputs:**

    * **marginals** (`list`):
        list of ``DistributionContinuous1D`` or ``DistributionDiscrete1D`` objects that define the marginals.

    Such a multivariate distribution possesses the following methods, on condition that all its univariate marginals
    also possess them:

    * ``pdf``, ``log_pdf``, ``cdf``, ``rvs``, ``fit``, ``moments``.

    The parameters of the distribution are only stored as attributes of the marginal objects. However, the
    *get_params* and *update_params* method can still be used for the joint. Note that, for this purpose, each parameter
    of the joint is assigned a unique string identifier as `key_index` - where `key` is the parameter name and `index`
    the index of the marginal (e.g., location parameter of the 2nd marginal is identified as `loc_1`).

    """
    def __init__(self, marginals):
        super().__init__()
        self.order_params = []
        for i, m in enumerate(marginals):
            self.order_params.extend([key + '_' + str(i) for key in m.order_params])

        # Check and save the marginals
        if not (isinstance(marginals, list) and all(isinstance(d, (DistributionContinuous1D, DistributionDiscrete1D))
                                                    for d in marginals)):
            raise ValueError('Input marginals must be a list of Distribution1d objects.')
        self.marginals = marginals

        # If all marginals have a method, the joint has it to
        if all(hasattr(m, 'pdf') or hasattr(m, 'pmf') for m in self.marginals):
            def joint_pdf(dist, x):
                x = dist._check_x_dimension(x)
                # Compute pdf of independent marginals
                pdf_val = np.ones((x.shape[0], ))
                for ind_m in range(len(self.marginals)):
                    if hasattr(self.marginals[ind_m], 'pdf'):
                        pdf_val *= marginals[ind_m].pdf(x[:, ind_m])
                    else:
                        pdf_val *= marginals[ind_m].pmf(x[:, ind_m])
                return pdf_val
            if any(hasattr(m, 'pdf') for m in self.marginals):
                self.pdf = MethodType(joint_pdf, self)
            else:
                self.pmf = MethodType(joint_pdf, self)

        if all(hasattr(m, 'log_pdf') or hasattr(m, 'log_pmf') for m in self.marginals):
            def joint_log_pdf(dist, x):
                x = dist._check_x_dimension(x)
                # Compute pdf of independent marginals
                pdf_val = np.zeros((x.shape[0],))
                for ind_m in range(len(self.marginals)):
                    if hasattr(self.marginals[ind_m], 'log_pdf'):
                        pdf_val += marginals[ind_m].log_pdf(x[:, ind_m])
                    else:
                        pdf_val += marginals[ind_m].log_pmf(x[:, ind_m])
                return pdf_val
            if any(hasattr(m, 'log_pdf') for m in self.marginals):
                self.log_pdf = MethodType(joint_log_pdf, self)
            else:
                self.log_pmf = MethodType(joint_log_pdf, self)

        if all(hasattr(m, 'cdf') for m in self.marginals):
            def joint_cdf(dist, x):
                x = dist._check_x_dimension(x)
                # Compute cdf of independent marginals
                cdf_val = np.prod(np.array([marg.cdf(x[:, ind_m])
                                            for ind_m, marg in enumerate(dist.marginals)]), axis=0)
                return cdf_val
            self.cdf = MethodType(joint_cdf, self)

        if all(hasattr(m, 'rvs') for m in self.marginals):
            def joint_rvs(dist, nsamples=1, random_state=None):
                # Go through all marginals
                rv_s = np.zeros((nsamples, len(dist.marginals)))
                for ind_m, marg in enumerate(dist.marginals):
                    rv_s[:, ind_m] = marg.rvs(nsamples=nsamples, random_state=random_state).reshape((-1,))
                return rv_s
            self.rvs = MethodType(joint_rvs, self)

        if all(hasattr(m, 'fit') for m in self.marginals):
            def joint_fit(dist, data):
                data = dist._check_x_dimension(data)
                # Compute ml estimates of independent marginal parameters
                mle_all = {}
                for ind_m, marg in enumerate(dist.marginals):
                    mle_i = marg.fit(data[:, ind_m])
                    mle_all.update({key+'_'+str(ind_m): val for key, val in mle_i.items()})
                return mle_all
            self.fit = MethodType(joint_fit, self)

        if all(hasattr(m, 'moments') for m in self.marginals):
            def joint_moments(dist, moments2return='mvsk'):
                # Go through all marginals
                if len(moments2return) == 1:
                    return np.array([marg.moments(moments2return=moments2return) for marg in dist.marginals])
                moments_ = [np.empty((len(dist.marginals), )) for _ in range(len(moments2return))]
                for ind_m, marg in enumerate(dist.marginals):
                    moments_i = marg.moments(moments2return=moments2return)
                    for j in range(len(moments2return)):
                        moments_[j][ind_m] = moments_i[j]
                return tuple(moments_)
            self.moments = MethodType(joint_moments, self)

    def get_params(self):
        """
        Return the parameters of a ``Distributions`` object.

        To update the parameters of a ``JointInd`` or a ``JointCopula`` distribution, each parameter is assigned a
        unique string identifier as `key_index` - where `key` is the parameter name and `index` the index of the
        marginal (e.g., location parameter of the 2nd marginal is identified as `loc_1`).

        **Output/Returns:**

        * (`dict`):
            Parameters of the distribution

        """
        params = {}
        for i, m in enumerate(self.marginals):
            params_m = m.get_params()
            for key, value in params_m.items():
                params[key + '_' + str(i)] = value
        return params

    def update_params(self, **kwargs):
        """
        Update the parameters of a ``Distributions`` object.

        To update the parameters of a ``JointInd`` or a ``JointCopula`` distribution, each parameter is assigned a
        unique string identifier as `key_index` - where `key` is the parameter name and `index` the index of the
        marginal (e.g., location parameter of the 2nd marginal is identified as `loc_1`).

        **Input:**

        * keyword arguments:
            Parameters to be updated

        """
        # check arguments
        all_keys = self.get_params().keys()
        # update the marginal parameters
        for key_indexed, value in kwargs.items():
            if key_indexed not in all_keys:
                raise ValueError('Unrecognized keyword argument ' + key_indexed)
            key_split = key_indexed.split('_')
            key, index = '_'.join(key_split[:-1]), int(key_split[-1])
            self.marginals[index].params[key] = value


class JointCopula(DistributionND):
    """
    Define a joint distribution from a list of marginals and a copula to introduce dependency. ``JointCopula`` is a
    child class of ``DistributionND``.

    **Inputs:**

    * **marginals** (`list`):
        `list` of ``DistributionContinuous1D`` or ``DistributionDiscrete1D`` objects that define the marginals

    * **copula** (`object`):
        object of class ``Copula``

    A ``JointCopula`` distribution may possess a ``cdf``, ``pdf`` and ``log_pdf`` methods if the copula allows for it
    (i.e., if the copula possesses the necessary ``evaluate_cdf`` and ``evaluate_pdf`` methods).

    The parameters of the distribution are only stored as attributes of the marginals/copula objects. However, the
    ``get_params`` and ``update_params`` methods can still be used for the joint. Note that each parameter of the joint
    is assigned a unique string identifier as `key_index` - where `key` is the parameter name and `index` the index of
    the marginal (e.g., location parameter of the 2nd marginal is identified as `loc_1`); and `key_c` for copula
    parameters.

    """
    def __init__(self, marginals, copula):
        super().__init__()
        self.order_params = []
        for i, m in enumerate(marginals):
            self.order_params.extend([key + '_' + str(i) for key in m.order_params])
        self.order_params.extend([key + '_c' for key in copula.order_params])

        # Check and save the marginals
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
                x = dist._check_x_dimension(x)
                # Compute cdf of independent marginals
                unif = np.array([marg.cdf(x[:, ind_m]) for ind_m, marg in enumerate(dist.marginals)]).T
                # Compute copula
                cdf_val = dist.copula.evaluate_cdf(unif=unif)
                return cdf_val
            self.cdf = MethodType(joint_cdf, self)

        if all(hasattr(m, 'pdf') for m in self.marginals) and hasattr(self.copula, 'evaluate_pdf'):
            def joint_pdf(dist, x):
                x = dist._check_x_dimension(x)
                # Compute pdf of independent marginals
                pdf_val = np.prod(np.array([marg.pdf(x[:, ind_m])
                                            for ind_m, marg in enumerate(dist.marginals)]), axis=0)
                # Add copula term
                unif = np.array([marg.cdf(x[:, ind_m]) for ind_m, marg in enumerate(dist.marginals)]).T
                c_ = dist.copula.evaluate_pdf(unif=unif)
                return c_ * pdf_val
            self.pdf = MethodType(joint_pdf, self)

        if all(hasattr(m, 'log_pdf') for m in self.marginals) and hasattr(self.copula, 'evaluate_pdf'):
            def joint_log_pdf(dist, x):
                x = dist._check_x_dimension(x)
                # Compute pdf of independent marginals
                logpdf_val = np.sum(np.array([marg.log_pdf(x[:, ind_m])
                                              for ind_m, marg in enumerate(dist.marginals)]), axis=0)
                # Add copula term
                unif = np.array([marg.cdf(x[:, ind_m]) for ind_m, marg in enumerate(dist.marginals)]).T
                c_ = dist.copula.evaluate_pdf(unif=unif)
                return np.log(c_) + logpdf_val
            self.log_pdf = MethodType(joint_log_pdf, self)

    def get_params(self):
        """
        Return the parameters of a ``Distributions`` object.

        To update the parameters of a ``JointInd`` or a ``JointCopula`` distribution, each parameter is assigned a
        unique string identifier as `key_index` - where `key` is the parameter name and `index` the index of the
        marginal (e.g., location parameter of the 2nd marginal is identified as `loc_1`).

        **Output/Returns:**

        * (`dict`):
            Parameters of the distribution.

        """
        params = {}
        for i, m in enumerate(self.marginals):
            for key, value in m.get_params().items():
                params[key + '_' + str(i)] = value
        for key, value in self.copula.get_params().items():
            params[key + '_c'] = value
        return params

    def update_params(self, **kwargs):
        """
        Update the parameters of a ``Distributions`` object.

        To update the parameters of a ``JointInd`` or a ``JointCopula`` distribution, each parameter is assigned a
        unique string identifier as `key_index` - where `key` is the parameter name and `index` the index of the
        marginal (e.g., location parameter of the 2nd marginal is identified as `loc_1`).

        **Input:**

        * keyword arguments:
            Parameters to be updated

        """
        # check arguments
        all_keys = self.get_params().keys()
        # update the marginal parameters
        for key_indexed, value in kwargs.items():
            if key_indexed not in all_keys:
                raise ValueError('Unrecognized keyword argument ' + key_indexed)
            key_split = key_indexed.split('_')
            key, index = '_'.join(key_split[:-1]), key_split[-1]
            if index == 'c':
                self.copula.params[key] = value
            else:
                self.marginals[int(index)].params[key] = value


########################################################################################################################
#        Copulas
########################################################################################################################

class Copula:
    """
    Define a copula for a multivariate distribution whose dependence structure is defined with a copula.

    This class is used in support of the ``JointCopula`` distribution class.

    **Attributes:**

    * **params** (`dict`):
        Parameters of the copula.

    * **order_params** (`list`):
        List of parameter names

    **Methods:**

    **evaluate_cdf** *(unif)*
        Compute the copula cdf :math:`C(u_1, u_2, ..., u_d)` for a `d`-variate uniform distribution.

        For a generic multivariate distribution with marginal cdfs :math:`F_1, ..., F_d` the joint cdf is computed as:

        :math:`F(x_1, ..., x_d) = C(u_1, u_2, ..., u_d)`

        where :math:`u_i = F_i(x_i)` is uniformly distributed. This computation is performed in the ``JointCopula.cdf``
        method.

        **Input:**

        * **unif** (`ndarray`):
            Points (uniformly distributed) at which to evaluate the copula cdf, must be of shape `(npoints, dimension)`.

        **Output/Returns:**

        * (`tuple`):
            Values of the cdf, `ndarray` of shape `(npoints, )`.

    **evaluate_pdf** *(unif)*
        Compute the copula pdf :math:`c(u_1, u_2, ..., u_d)` for a `d`-variate uniform distribution.

        For a generic multivariate distribution with marginals pdfs :math:`f_1, ..., f_d` and marginals cdfs
        :math:`F_1, ..., F_d`, the joint pdf is computed as:

        :math:`f(x_1, ..., x_d) = c(u_1, u_2, ..., u_d) f_1(x_1) ... f_d(x_d)`

        where :math:`u_i = F_i(x_i)` is uniformly distributed. This computation is performed in the ``JointCopula.pdf``
        method.

        **Input:**

        * **unif** (`ndarray`):
            Points (uniformly distributed) at which to evaluate the copula pdf, must be of shape `(npoints, dimension)`.

        **Output/Returns:**

        * (`tuple`):
            Values of the copula pdf term, ndarray of shape `(npoints, )`.

    """
    def __init__(self, order_params=None, **kwargs):
        self.params = kwargs
        self.order_params = order_params
        if self.order_params is None:
            self.order_params = tuple(kwargs.keys())
        if len(self.order_params) != len(self.params):
            raise ValueError('Inconsistent dimensions between order_params tuple and params dictionary.')

    def check_marginals(self, marginals):
        """
        Perform some checks on the marginals, raise errors if necessary.

        As an example, Archimedian copula are only defined for bi-variate continuous distributions, thus this method
        checks that marginals is of length 2 and continuous, and raise an error if that is not the case.

        **Input:**

        * **unif** (ndarray):
            Points (uniformly distributed) at which to evaluate the copula pdf, must be of shape
            ``(npoints, dimension)``.

        **Output/Returns:**

        No outputs, this code raises errors if necessary.

        """
        pass

    def get_params(self):
        return self.params

    def update_params(self, **kwargs):
        for key in kwargs.keys():
            if key not in self.params.keys():
                raise ValueError('Wrong parameter name.')
            self.params[key] = kwargs[key]


class Gumbel(Copula):
    """
    Gumbel copula having cumulative distribution function

    .. math:: F(u_1, u_2) = \exp(-(-\log(u_1))^{\Theta} + (-\log(u_2))^{\Theta})^{1/{\Theta}}

    where :math:`u_1 = F_1(x_1), u_2 = F_2(x_2)` are uniformly distributed on the interval `[0, 1]`.

    **Input:**

    * **theta** (`float`):
        Parameter of the Gumbel copula, real number in :math:`[1, +\infty)`.

    This copula possesses the following methods:

    * ``evaluate_cdf``, ``evaluate_pdf`` and ``check_copula``

    (``check_copula`` checks that `marginals` consist of solely 2 continuous univariate distributions).
    """
    def __init__(self, theta):
        # Check the input copula_params
        if theta is not None and ((not isinstance(theta, (float, int))) or (theta < 1)):
            raise ValueError('Input theta should be a float in [1, +oo).')
        super().__init__(theta=theta)

    def evaluate_cdf(self, unif):
        if unif.shape[1] > 2:
            raise ValueError('Maximum dimension for the Gumbel Copula is 2.')
        if self.params['theta'] == 1:
            return np.prod(unif, axis=1)

        u = unif[:, 0]
        v = unif[:, 1]
        theta = self.params['theta']
        cdf_val = np.exp(-((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta))

        return cdf_val

    def evaluate_pdf(self, unif):
        if unif.shape[1] > 2:
            raise ValueError('Maximum dimension for the Gumbel Copula is 2.')
        if self.params['theta'] == 1:
            return np.ones(unif.shape[0])

        u = unif[:, 0]
        v = unif[:, 1]
        theta = self.params['theta']
        c = np.exp(-((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1 / theta))

        pdf_val = c * 1 / u * 1 / v * ((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (-2 + 2 / theta) \
             * (np.log(u) * np.log(v)) ** (theta - 1) * \
             (1 + (theta - 1) * ((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (-1 / theta))
        return pdf_val

    def check_marginals(self, marginals):
        """
        Check that marginals contains 2 continuous univariate distributions.
        """
        if len(marginals) != 2:
            raise ValueError('Maximum dimension for the Gumbel Copula is 2.')
        if not all(isinstance(m, DistributionContinuous1D) for m in marginals):
            raise ValueError('Marginals should be 1d continuous distributions.')


class Clayton(Copula):
    """
    Clayton copula having cumulative distribution function

    .. math:: F(u_1, u_2) = \max(u_1^{-\Theta} + u_2^{-\Theta} - 1, 0)^{-1/{\Theta}}

    where :math:`u_1 = F_1(x_1), u_2 = F_2(x_2)` are uniformly distributed on the interval `[0, 1]`.

    **Input:**

    * **theta** (`float`):
        Parameter of the copula, real number in [-1, +oo)\{0}.

    This copula possesses the following methods:

    * ``evaluate_cdf`` and ``check_copula``

    (``check_copula`` checks that `marginals` consist of solely 2 continuous univariate distributions).
    """
    def __init__(self, theta):
        # Check the input copula_params
        if theta is not None and ((not isinstance(theta, (float, int))) or (theta < -1 or theta == 0.)):
            raise ValueError('Input theta should be a float in [-1, +oo)\{0}.')
        super().__init__(theta=theta)

    def evaluate_cdf(self, unif):
        if unif.shape[1] > 2:
            raise ValueError('Maximum dimension for the Clayton Copula is 2.')
        if self.params['theta'] == 1:
            return np.prod(unif, axis=1)

        u = unif[:, 0]
        v = unif[:, 1]
        theta = self.params['theta']
        cdf_val = (np.maximum(u ** (-theta) + v ** (-theta) - 1., 0.)) ** (-1. / theta)
        return cdf_val

    def check_marginals(self, marginals):
        if len(marginals) != 2:
            raise ValueError('Maximum dimension for the Clayton Copula is 2.')
        if not all(isinstance(m, DistributionContinuous1D) for m in marginals):
            raise ValueError('Marginals should be 1d continuous distributions.')


class Frank(Copula):
    """
    Frank copula having cumulative distribution function

    :math:`F(u_1, u_2) = -\dfrac{1}{\Theta} \log(1+\dfrac{(\exp(-\Theta u_1)-1)(\exp(-\Theta u_2)-1)}{\exp(-\Theta)-1})`

    where :math:`u_1 = F_1(x_1), u_2 = F_2(x_2)` are uniformly distributed on the interval `[0, 1]`.

    **Input:**

    * **theta** (`float`):
        Parameter of the copula, real number in correlation_function\{0}.

    This copula possesses the following methods:

    * ``evaluate_cdf`` and ``check_copula``

    (``check_copula`` checks that `marginals` consist of solely 2 continuous univariate distributions).
    """
    def __init__(self, theta):
        # Check the input copula_params
        if theta is not None and ((not isinstance(theta, (float, int))) or (theta == 0.)):
            raise ValueError('Input theta should be a float in correlation_function\{0}.')
        super().__init__(theta=theta)

    def evaluate_cdf(self, unif):
        if unif.shape[1] > 2:
            raise ValueError('Maximum dimension for the Clayton Copula is 2.')
        if self.params['theta'] == 1:
            return np.prod(unif, axis=1)

        u = unif[:, 0]
        v = unif[:, 1]
        theta = self.params['theta']
        tmp_ratio = (np.exp(-theta * u) - 1.) * (np.exp(-theta * v) - 1.) / (np.exp(-theta) - 1.)
        cdf_val = -1. / theta * np.log(1. + tmp_ratio)
        return cdf_val

    def check_marginals(self, marginals):
        if len(marginals) != 2:
            raise ValueError('Maximum dimension for the Frank Copula is 2.')
        if not all(isinstance(m, DistributionContinuous1D) for m in marginals):
            raise ValueError('Marginals should be 1d continuous distributions.')

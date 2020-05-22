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

from .Utilities import check_input_dims


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

    **moments** *()*
        Compute the first four non-central moments of a distribution.

        For a univariate distribution, mean, variance, skewness and kurtosis are returned. For a multivariate
        distribution, the mean vector, covariance and vectors of marginal skewness and marginal kurtosis are returned.
        If a given moment cannot be computed, None is returned in its place.

        **Output/Returns:**

        * (`tuple`):
            ``mean``: mean, ``var``:  covariance, ``skew``: skewness, ``kurt``: kurtosis.

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
        Update the parameters of the ``Distribution`` object.

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
        Return the parameters of the distribution object.

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
        self.moments = lambda: scipy_name.stats(moments='mvsk', **self.params)
        self.rvs = lambda nsamples, random_state=None: scipy_name.rvs(
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
        super().__init__(ploc=loc, scale=scale, order_params=('loc', 'scale'))
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

    >>> from UQpy.Distributions import Binomial
    >>> dist = Binomial(n=5, p=0.4)
    >>> print(Binomial.__bases__)
    (<class 'UQpy.Distributions.ScipyDiscrete'>,)
    >>> print(dist.rvs(nsamples=3, random_state=123))
    [[3]
     [1]
     [1]]
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
        self.moments = lambda: scipy_name.stats(moments='mvsk', **self.params)
        self.rvs = lambda nsamples, random_state=None: scipy_name.rvs(
            size=nsamples, random_state=random_state, **self.params).reshape((nsamples, 1))


class Binomial(DistributionDiscrete1D):
    """
    Binomial distribution

    **Inputs:**

    * **n** (`int`):
        number of trials, integer >= 0
    * **p** (`float`):
        success probability for each trial, real number in [0, 1]
    * **loc** (`float`):
        location parameter

    The following methods are available for Binomial: `cdf, pmf, log_pmf, icdf, rvs, moments`.
    """
    def __init__(self, n, p, loc=0.):
        super().__init__(n=n, p=p, loc=loc, order_params=('n', 'p', 'loc'))
        self._construct_from_scipy(scipy_name=stats.binom)


class Poisson(DistributionDiscrete1D):
    """
    Poisson distribution

    **Inputs:**

    * **mu** (`float`):
        shape parameter
    * **loc** (`float`):
        location parameter

    The following methods are available for Poisson: `cdf, pmf, log_pmf, icdf, rvs, moments`.
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

    >>> from UQpy.Distributions import MVNormal
    >>> print(MVNormal.__bases__)
        (<class 'UQpy.Distributions.DistributionND'>,)
    >>> dist = MVNormal(mean=[1., 2.], cov=[[4., -0.2], [-0.2, 1.]])
    >>> print(dist.rvs(nsamples=5, random_state=123))
        [[ 3.23569787  2.84449354]
         [ 0.33525582  0.54456529]
         [ 2.26521608  3.56007209]
         [ 5.82251567  1.25292155]
         [-1.58752203  1.30887881]]
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
    Multivariate normal

    **Inputs:**

    * **mean** (`ndarray`):
        mean vector, `ndarray` of shape (dimension, )
    * **cov** (`float` or `ndarray`):
        covariance, `float` or `ndarray` of shape (dimension, ) or (dimension, dimension). Default is 1.

    The following methods are available for MVNormal: `pdf, log_pdf, rvs, fit`.
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


class Multinomial(DistributionND):
    """
    Multinomial distribution

    **Inputs:**

    * **n** (`int`):
        number of trials
    * **p** (`array_like`):
        probability of a trial falling into each category; should sum to 1

    The following methods are available for MVNormal: `pmf, log_pmf, rvs`.
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


class JointInd(DistributionND):
    """
    Define a joint distribution from its independent marginals.

    **Inputs:**

    * **marginals** (`list`):
        list of ``DistributionContinuous1D`` or ``DistributionDiscrete1D`` objects that define the marginals.

    Such a multivariate distribution possesses the following methods, on condition that all its univariate marginals
    also possess them: `pdf, log_pdf, cdf, rvs, fit, moments`.

    The parameters of the distribution are only stored as attributes of the marginal objects. However, the
    *get_params* and *update_params* method can still be used for the joint. Note that for this puspose each parameter
    of the joint is assigned a unique string identifier as key_index - where key is the parameter name and index the
    index of the marginal (e.g., location parameter of the 2nd marginal is identified as loc_1).

    >>> from UQpy.Distributions import Normal, Lognormal, JointInd
    >>> marginals = [Normal(loc=2., scale=2.), Lognormal(s=1., loc=0., scale=np.exp(5))]
    >>> dist = JointInd(marginals=marginals)
    >>> print(dist.rvs(nsamples=3, random_state=123))
    [[-1.71261207e-01  5.01174573e+01]
     [ 3.99469089e+00  4.02359290e+02]
     [ 2.56595700e+00  1.96955635e+02]]
    >>> print([m.params for m in marginals])
    [{'loc': 2.0, 'scale': 2.0}, {'s': 1.0, 'loc': 0.0, 'scale': 148.4131591025766}]
    >>> dist.update_params(loc_1=1.)
    >>> print([m.params for m in marginals])
    [{'loc': 2.0, 'scale': 2.0}, {'s': 1.0, 'loc': 1.0, 'scale': 148.4131591025766}]

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
                x = dist._check_x_dimension(x)
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
                x = dist._check_x_dimension(x)
                # Compute cdf of independent marginals
                cdf_val = np.prod(np.array([m.cdf(x[:, i]) for i, m in enumerate(dist.marginals)]), axis=0)
                return cdf_val
            self.cdf = MethodType(joint_cdf, self)

        if all(hasattr(m, 'rvs') for m in self.marginals):
            def joint_rvs(dist, nsamples=1, random_state=None):
                # Go through all marginals
                rv_s = np.zeros((nsamples, len(dist.marginals)))
                for i, m in enumerate(dist.marginals):
                    rv_s[:, i] = m.rvs(nsamples=nsamples, random_state=random_state).reshape((-1,))
                return rv_s
            self.rvs = MethodType(joint_rvs, self)

        if all(hasattr(m, 'fit') for m in self.marginals):
            def joint_fit(dist, data):
                data = dist._check_x_dimension(data)
                # Compute ml estimates of independent marginal parameters
                mle_all = {}
                for i, m in enumerate(dist.marginals):
                    mle_i = m.fit(data[:, i])
                    mle_all.update({key+'_'+str(i): val for key, val in mle_i.items()})
                return mle_all
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

    def get_params(self):
        """
        Return the parameters of the joint distribution.

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
        Update the parameters of the marginals.

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
    Define a joint distribution from a list of marginals and a copula to introduce dependency.

    **Inputs:**

    * **marginals** (`list`):
        `list` of ``DistributionContinuous1D`` or ``DistributionDiscrete1D`` objects that define the marginals

    * **copula** (`object`):
        object of class ``Copula``

    Such a multivariate distribution may possess a `cdf, pdf` and `log_pdf` methods if the copula allows for it (i.e.,
    if the copula possesses the necessary `evaluate_cdf` and `evaluate_pdf` methods).

    The parameters of the distribution are only stored as attributes of the marginals/copula objects. However, the
    `get_params` and `update_params` method can still be used for the joint. Note that each parameter of the joint is
    assigned a unique string identifier as key_index - where key is the parameter name and index the index of the
    marginal (e.g., location parameter of the 2nd marginal is identified as loc_1); and key_c for copula parameters.

    >>> from UQpy.Distributions import JointCopula, Normal, Gumbel
    >>> marginals = [Normal(loc=0., scale=1), Normal(loc=0., scale=1)]
    >>> copula = Gumbel(theta=3.)
    >>> dist = JointCopula(marginals=marginals, copula=copula)
    >>> print(dist.get_params())
    {'loc_0': 0., 'scale_0': 1., 'loc_1': 0., 'scale_1': 1., 'theta_c': 3.}
    >>> print(hasattr(dist, 'rvs'))
    False
    >>> print(dist.copula.params)
    {'theta': 3.0}
    >>> dist.update_params(theta_c=2.)
    >>> print(dist.copula.params)
    {'theta': 2.0}
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
                unif = np.array([m.cdf(x[:, i]) for i, m in enumerate(dist.marginals)]).T
                # Compute copula
                cdf_val = dist.copula.evaluate_cdf(unif=unif)
                return cdf_val
            self.cdf = MethodType(joint_cdf, self)

        if all(hasattr(m, 'pdf') for m in self.marginals) and hasattr(self.copula, 'evaluate_pdf'):
            def joint_pdf(dist, x):
                x = dist._check_x_dimension(x)
                # Compute pdf of independent marginals
                pdf_val = np.prod(np.array([m.pdf(x[:, i]) for i, m in enumerate(dist.marginals)]), axis=0)
                # Add copula term
                unif = np.array([m.cdf(x[:, i]) for i, m in enumerate(dist.marginals)]).T
                c_ = dist.copula.evaluate_pdf(unif=unif)
                return c_ * pdf_val
            self.pdf = MethodType(joint_pdf, self)

        if all(hasattr(m, 'log_pdf') for m in self.marginals) and hasattr(self.copula, 'evaluate_pdf'):
            def joint_log_pdf(dist, x):
                x = dist._check_x_dimension(x)
                # Compute pdf of independent marginals
                logpdf_val = np.sum(np.array([m.log_pdf(x[:, i]) for i, m in enumerate(dist.marginals)]), axis=0)
                # Add copula term
                unif = np.array([m.cdf(x[:, i]) for i, m in enumerate(dist.marginals)]).T
                c_ = dist.copula.evaluate_pdf(unif=unif)
                return np.log(c_) + logpdf_val
            self.log_pdf = MethodType(joint_log_pdf, self)

    def get_params(self):
        """
        Return the parameters of the joint distribution.

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
        Update the parameters of the marginals and/or copula.

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

    This class is used in support of the JointCopula distribution class.

    **Attributes:**

    * **params** (`dict`):
        Parameters of the copula.

    * **order_params** (`list`):
        List of parameter names

    **Methods:**

    **evaluate_cdf** *(unif)*
        Compute the copula cdf C(u1, u2, ..., ud) for a d-variate uniform distribution.

        For a generic multivariate distribution with marginal cdfs F1, ...Fd, the joint cdf is computed as:

        F(x1, ..., xd) = C(u1, u2, ..., ud)

        where ui = Fi(xi) is uniformly distributed. This computation is performed in the JointCopula.cdf method.

        **Input:**

        * **unif** (`ndarray`):
            Points (uniformly distributed) at which to evaluate the copula cdf, must be of shape
            ``(npoints, dimension)``.

        **Output/Returns:**

        * (`tuple`):
            Values of the cdf, `ndarray` of shape ``(npoints, )``.

    **evaluate_pdf** *(unif)*
        Compute the copula pdf term c(u1, u2, ..., ud) for a d-variate uniform distribution.

        For a generic multivariate distribution with marginals pdfs f1, ..., fd and marginals cdfs F1, ...Fd, the
        joint pdf is computed as:

        f(x1, ..., xd) = c(u1, u2, ..., ud) * f1(x1) * ... * fd(xd)

        where ui = Fi(xi) is uniformly distributed. This computation is performed in the JointCopula.pdf method.

        **Input:**

        * **unif** (`ndarray`):
            Points (uniformly distributed) at which to evaluate the copula pdf, must be of shape
            ``(npoints, dimension)``.

        **Output/Returns:**

        * (`tuple`):
            Values of the copula pdf term, ndarray of shape ``(npoints, )``.

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
        Perform some checks on the marginals, raise Errors if necessary.

        As an example, Archimedian copula are only defined for bi-variate continuous distributions, thus this method
        would check that marginals is of length 2 and continuous, and raise an error if that is not the case.

        **Input:**

        * **unif** (ndarray):
            Points (uniformly distributed) at which to evaluate the copula pdf, must be of shape
            ``(npoints, dimension)``.

        **Output/Returns:**

        No outputs, this code raises Errors if necessary.

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
    Gumbel copula

    **Input:**

    * **theta** (`float`):
        Parameter of the Gumbel copula, real number in [1, +oo).

    This copula possesses the following methods: `evaluate_cdf, evaluate_pdf` and `check_copula` (the latter checks
    that `marginals` consist of solely 2 continuous distributions).
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
    Clayton copula

    **Input:**

    * **theta** (`float`):
        Parameter of the copula, real number in [-1, +oo)\{0}.

    This copula possesses the following methods: `evaluate_cdf` and `check_copula` (the latter checks
    that `marginals` consist of solely 2 continuous distributions).
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
    Frank copula

    **Input:**

    * **theta** (`float`):
        Parameter of the copula, real number in R\{0}.

    This copula possesses the following methods: `evaluate_cdf` and `check_copula` (the latter checks
    that `marginals` consist of solely 2 continuous distributions).
    """
    def __init__(self, theta):
        # Check the input copula_params
        if theta is not None and ((not isinstance(theta, (float, int))) or (theta == 0.)):
            raise ValueError('Input theta should be a float in R\{0}.')
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
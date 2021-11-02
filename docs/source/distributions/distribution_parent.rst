Parent Distribution Class
----------------------------------

.. autoclass:: UQpy.distributions.baseclass.Distribution
    :members:
    :private-members:

**Additional methods available from :py:mod:`scipy.stats` :**
    **cdf** *(x)*
        Evaluate the cumulative distribution function.

        :param numpy.ndarray x: Point(s) at which to evaluate the `cdf`, must be of shape `(npoints, dimension)`.
        :return: Evaluated cdf values, `ndarray` of shape `(npoints,)`.
        :rtype: numpy.ndarray
    **pdf** *(x)*
        Evaluate the probability density function of a continuous or multivariate mixed continuous-discrete
        distribution.

        :param numpy.ndarray x: Point(s) at which to evaluate the `pdf`, must be of shape `(npoints, dimension)`.
        :return: Evaluated pdf values, `ndarray` of shape `(npoints,)`.
        :rtype: numpy.ndarray

    **pmf** *(x)*
        Evaluate the probability mass function of a discrete distribution.

        :param numpy.ndarray x: Point(s) at which to evaluate the `pmf`, must be of shape `(npoints, dimension)`.
        :return: Evaluated pmf values, `ndarray` of shape `(npoints,)`.
        :rtype: numpy.ndarray

    **log_pdf** *(x)*
        Evaluate the logarithm of the probability density function of a continuous or multivariate mixed
        continuous-discrete distribution.

        :param numpy.ndarray x: Point(s) at which to evaluate the `log_pdf`, must be of shape `(npoints, dimension)`.
        :return: Evaluated log-pdf values, `ndarray` of shape `(npoints,)`.
        :rtype: numpy.ndarray

    **log_pmf** *(x)*
        Evaluate the logarithm of the probability mass function of a discrete distribution.

        :param numpy.ndarray x: Point(s) at which to evaluate the `log_pmf`, must be of shape `(npoints, dimension)`.
        :return: Evaluated log-pmf values, `ndarray` of shape `(npoints,)`.
        :rtype: numpy.ndarray

    **icdf** *(x)*
        Evaluate the inverse cumulative distribution function for univariate distributions.

        :param numpy.ndarray x: Point(s) at which to evaluate the `icdf`, must be of shape `(npoints, dimension)`.
        :return: Evaluated icdf values, `ndarray` of shape `(npoints,)`.
        :rtype: numpy.ndarray

    **rvs** *(nsamples=1, random_state=None)*
        Sample independent identically distributed (iid) realizations.

        :param int nsamples: Number of iid samples to be drawn. Default is 1.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is None.
         If an integer is provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise, the
         object itself can be passed directly.
        :return: Generated iid samples, `ndarray` of shape `(npoints, dimension)`.
        :rtype: numpy.ndarray

    **moments** *(moments2return='mvsk')*
        Computes the mean ('m'), variance/covariance ('v'), skewness ('s') and/or kurtosis ('k') of the distribution.
        For a univariate distribution, mean, variance, skewness and kurtosis are returned. For a multivariate
        distribution, the mean vector, covariance and vectors of marginal skewness and marginal kurtosis are returned.

        :param str moments2return: Indicates which moments are to be returned (mean, variance, skewness and/or kurtosis). Default is 'mvsk'.
        :return: ``mean``: mean, ``var``:  variance/covariance, ``skew``: skewness, ``kurt``: kurtosis.
        :rtype: tuple

    **fit** *(data)*
        Compute the maximum-likelihood parameters from iid data.
        Computes the mle analytically if possible. For univariate continuous distributions, it leverages the fit
        method of the scipy.stats package.

        :param numpy.ndarray data: Data array, must be of shape `(npoints, dimension)`.
        :return: Maximum-likelihood parameter estimates.
        :rtype: dict

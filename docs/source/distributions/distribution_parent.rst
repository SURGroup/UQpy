Parent Distribution Class
----------------------------------

Methods
~~~~~~~~~~~~~~~~~~
.. autoclass:: UQpy.distributions.baseclass.Distribution
    :members: update_parameters, get_parameters

Attributes
~~~~~~~~~~~~~~~~~~
.. autoattribute:: UQpy.distributions.baseclass.Distribution.parameters
.. autoattribute:: UQpy.distributions.baseclass.Distribution.ordered_parameters


Additional methods available from :py:mod:`scipy.stats` :
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    .. py:function:: cdf(x)

        Evaluate the cumulative distribution function.

        :param x: Point(s) at which to evaluate the :py:meth:`cdf`, must be :class:`numpy.ndarray` of shape :code:`(npoints, dimension)`.
        :return: Evaluated cdf values, :class:`numpy.ndarray` of shape :code:`(npoints,)`.
    .. py:function:: pdf(x)

        Evaluate the probability density function of a continuous or multivariate mixed continuous-discrete
        distribution.

        :param x: Point(s) at which to evaluate the :py:meth:`pdf`, must be :class:`numpy.ndarray` of shape :code:`(npoints, dimension)`.
        :return: Evaluated pdf values, :class:`numpy.ndarray` of shape :code:`(npoints,)`.

    .. py:function:: pmf(x)

        Evaluate the probability mass function of a discrete distribution.

        :param x: Point(s) at which to evaluate the :py:meth:`pmf`, must be :class:`numpy.ndarray` of shape :code:`(npoints, dimension)`.
        :return: Evaluated pmf values, :class:`numpy.ndarray` of shape :code:`(npoints,)`.

    .. py:function:: log_pdf(x)

        Evaluate the logarithm of the probability density function of a continuous or multivariate mixed
        continuous-discrete distribution.

        :param x: Point(s) at which to evaluate the :py:meth:`log_pdf`, must be :class:`numpy.ndarray` of shape :code:`(npoints, dimension)`.
        :return: Evaluated log-pdf values, :class:`numpy.ndarray` of shape :code:`(npoints,)`.

    .. py:function:: log_pmf(x)

        Evaluate the logarithm of the probability mass function of a discrete distribution.

        :param x: Point(s) at which to evaluate the :py:meth:`log_pmf`, must be :class:`numpy.ndarray` of shape :code:`(npoints, dimension)`.
        :return: Evaluated log-pmf values, :class:`numpy.ndarray` of shape :code:`(npoints,)`.

    .. py:function:: icdf(x)

        Evaluate the inverse cumulative distribution function for univariate distributions.

        :param x: Point(s) at which to evaluate the :py:meth:`icdf`, must be of shape :code:`(npoints, dimension)`.
        :return: Evaluated icdf values, :class:`numpy.ndarray` of shape :code:`(npoints,)`.
        :rtype: numpy.ndarray

    .. py:function:: rvs(nsamples=1, random_state=None)

        Sample independent identically distributed (iid) realizations.

        :param nsamples: Number of iid samples to be drawn. Default is 1.
        :param random_state: Random seed used to initialize the pseudo-random number generator. Default is :any:`None`.
         If an :any:`int` is provided, this sets the seed for an object of :class:`numpy.random.RandomState`. Otherwise, the
         object itself can be passed directly.
        :return: Generated iid samples, `:class:`numpy.ndarray` of shape :code:`(npoints, dimension)`.

    .. py:function:: moments(moments2return='mvsk')

        Computes the mean ('m'), variance/covariance ('v'), skewness ('s') and/or kurtosis ('k') of the distribution.
        For a univariate distribution, mean, variance, skewness and kurtosis are returned. For a multivariate
        distribution, the mean vector, covariance and vectors of marginal skewness and marginal kurtosis are returned.

        :param moments2return: Indicates which moments are to be returned (mean, variance, skewness and/or kurtosis). Default is 'mvsk'.
        :return: ``mean``: mean, ``var``:  variance/covariance, ``skew``: skewness, ``kurt``: kurtosis.
        :rtype: tuple

    .. py:function:: fit(data)

        Compute the maximum-likelihood parameters from iid data.
        Computes the mle analytically if possible. For univariate continuous distributions, it leverages the fit
        method of the :py:mod:`scipy.stats` package.

        :param data: Data array, must be :class:`numpy.ndarray` of shape :code:`(npoints, dimension)`.
        :return: Maximum-likelihood parameter estimates.
        :rtype: dict

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

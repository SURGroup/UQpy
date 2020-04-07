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
from .Utilities import check_input_dims
import importlib
import types

########################################################################################################################
#        Define the probability distribution of the random parameters
########################################################################################################################


# The supported univariate distributions are:
list_univariates = ['normal', 'uniform', 'binomial', 'beta', 'genextreme', 'chisquare', 'lognormal', 'gamma',
                    'exponential', 'cauchy', 'levy', 'logistic', 'laplace', 'maxwell', 'inverse gauss', 'pareto',
                    'rayleigh', 'truncnorm']
# The supported multivariate distributions are:
list_multivariates = ['mvnormal']
# All scipy supported distributions
list_all_scipy = list_univariates + list_multivariates


class Distribution:
    """
    Main distribution class available to the user. The user can define a probability distribution by providing:
    - A name that points to a univariate/multivariate distribution (see supported distributions in
       SubDistribution class or custom distribution)
    - A list of names that points to a list of univariate distributions. In that case, a multivariate
       distribution is built for which all dimensions are independent and given by Distribution(name)
    - A list of names and a copula, in that case a multivariate distribution is built using Distribution(name)
        for the marginal pdfs, while the dependence structure is given by the copula.

    The following methods are defined:

    1. pdf: probability density function
    2. cdf: cumulative distribution function
    3. icdf: inverse cumulative distribution function
    4. rvs: generate random numbers (it doesn't need a point)
    5. log_pdf: logarithm of the pdf
    6. fit: Estimates the parameters of the distribution over arbitrary data
    7. moments: Calculate the first four moments of the distribution (mean, variance, skewness, kurtosis)

    **Input:**

    :param dist_name: Name of distribution.
    :type: dist_name: string or list of strings

    :param params: list of parameters for the distribution (list of lists if distribution is defined via its
                        marginals)
    :type: params: list of lists or ndarray

    :param copula: Copula to create dependence within dimensions, used only if name is a list
    :type: copula: str or None (default None)

    :param copula_params: Parameters of the copula
    :type: copula_params: list or ndarray

    **Attributes:**

    :param self.pdf: Probability density function
    :type: self.pdf: ndarray

    :param self.cdf: Cumulative distribution function
    :type: self.cdf: ndarray

    :param self.icdf: Inverse cumulative distribution function
    :type: self.icdf: ndarray

    :param self.rvs: Identical distributed realizations of the random variable
    :type: self.rvs: ndarray

    :param self.log_pdf: Logarithm of the pdf
    :type: self.log_pdf: ndarray

    :param self.fit: Estimates the parameters of the distribution over arbitrary data
    :type: self.fit: ndarray

    :param self.moments: Calculate the first four moments of the distribution (mean, variance, skewness, kurtosis)
    :type: self.moments: ndarray

    **Authors:**

    Dimitris Giovanis & Audrey Olivier
    """

    def __init__(self, dist_name, copula=None, params=None, copula_params=None):

        # Check dist_name
        if isinstance(dist_name, str):
            if not (dist_name.lower() in list_all_scipy or os.path.isfile(os.path.join(dist_name + '.py'))):
                raise ValueError('dist_name should be a supported density or name of an existing .py file')
        elif isinstance(dist_name, (list, tuple)) and all(isinstance(d_, str) for d_ in dist_name):
            if not all([(d_.lower() in list_all_scipy or os.path.isfile(os.path.join(d_ + '.py'))) for d_ in dist_name]):
                raise ValueError('dist_name should be a list of supported densities or names of an existing .py file')
        else:
            raise TypeError('dist_name should be a (list of) string(s)')
        self.dist_name = dist_name

        # Instantiate copula
        if copula is not None:
            if not isinstance(copula, str):
                raise ValueError('UQpy error: when provided, copula should be a string.')
            if isinstance(dist_name, str):
                raise ValueError('UQpy error: it does not make sense to define a copula when name is a single string.')
            self.copula = Copula(copula_name=copula, dist_name=self.dist_name)

        # Method that saves the parameters as attributes of the class if they are provided
        self.update_params(params, copula_params)

        # Other methods: you first need to check that they exist
        exist_methods = {}
        for method in ['pdf', 'log_pdf', 'cdf', 'rvs', 'icdf', 'fit', 'moments']:
            exist_methods[method] = exist_method(method=method, dist_name=self.dist_name,
                                                 has_copula=hasattr(self, 'copula'))
        if exist_methods['pdf']:
            self.pdf = types.MethodType(define_pdf, self)
        if exist_methods['log_pdf']:
            self.log_pdf = types.MethodType(define_log_pdf, self)
        if exist_methods['cdf']:
            self.cdf = types.MethodType(define_cdf, self)
        if exist_methods['icdf']:
            self.icdf = types.MethodType(define_icdf, self)
        if exist_methods['rvs']:
            self.rvs = types.MethodType(define_rvs, self)
        if exist_methods['fit']:
            self.fit = types.MethodType(define_fit, self)
        if exist_methods['moments']:
            self.moments = types.MethodType(define_moments, self)

    def update_params(self, params=None, copula_params=None):
        # Update the parameters, unless they are given as None, then do not update
        if params is not None:
            self.params = params
        if copula_params is not None:
            self.copula_params = copula_params


# Define the function that computes pdf
def define_pdf(self, x, params=None, copula_params=None):
    """
    A method that computes the probability density function at inputs points x
    
    **Input:**

    :param x: Points to estimate the pdf.
    :type x:  2D ndarray (nsamples, dimension)
                nsamples: an integer providing the desired number of iid samples to be drawn

    :param params: list of parameters for the distribution (list of lists if distribution is defined via its
                        marginals)
    :type: params: list of lists or ndarray

    :param copula_params: Parameters of the copula
    :type: copula_params: list or ndarray

    **Output:**

    :return prod_pdf: Values of the pdf
    :rtype prod_pdf: ndarray
    """
    x = check_input_dims(x)
    self.update_params(params, copula_params)
    if isinstance(self.dist_name, str):
        return subdistribution_pdf(dist_name=self.dist_name, x=x, params=self.params)
    elif isinstance(self.dist_name, list):
        if (x.shape[1] != len(self.dist_name)) or (len(self.params) != len(self.dist_name)):
            raise ValueError('Inconsistent dimensions in inputs dist_name and params.')
        prod_pdf = np.ones((x.shape[0], ))
        for i in range(len(self.dist_name)):
            prod_pdf = prod_pdf * subdistribution_pdf(dist_name=self.dist_name[i], x=x[:, i, np.newaxis],
                                                      params=self.params[i])
        if hasattr(self, 'copula'):
            _, c_ = self.copula.evaluate_copula(x=x, dist_params=self.params, copula_params=self.copula_params)
            prod_pdf *= c_
        return prod_pdf


# Define the function that computes the log pdf
def define_log_pdf(self, x, params=None, copula_params=None):
    """
    A method that computes the logarithmic probability density function at inputs points x

    **Input:**

    :param x: Points to estimate the pdf.
    :type x:  2D ndarray (nsamples, dimension)
                nsamples: an integer providing the desired number of iid samples to be drawn

    :param params: list of parameters for the distribution (list of lists if distribution is defined via its
                        marginals)
    :type: params: list of lists or ndarray

    :param copula_params: Parameters of the copula
    :type: copula_params: list or ndarray

    **Output:**

    :return sum_log_pdf: Values of the pdf
    :rtype sum_log_pdf: ndarray
    """
    x = check_input_dims(x)
    self.update_params(params, copula_params)
    if isinstance(self.dist_name, str):
        return subdistribution_log_pdf(dist_name=self.dist_name, x=x, params=self.params)
    elif isinstance(self.dist_name, list):
        if (x.shape[1] != len(self.dist_name)) or (len(self.params) != len(self.dist_name)):
            raise ValueError('Inconsistent dimensions in inputs dist_name and params.')
        sum_log_pdf = np.zeros((x.shape[0], ))
        for i in range(len(self.dist_name)):
            sum_log_pdf = sum_log_pdf + subdistribution_log_pdf(dist_name=self.dist_name[i], x=x[:, i, np.newaxis],
                                                                params=self.params[i])
        if hasattr(self, 'copula'):
            _, c_ = self.copula.evaluate_copula(x=x, dist_params=self.params, copula_params=self.copula_params)
            sum_log_pdf += np.log(c_)
        return sum_log_pdf


# Function that computes the cdf
def define_cdf(self, x, params=None, copula_params=None):
    """
    A method that computes the cumulative distribution function at inputs points x

    **Input:**

    :param x: Points to estimate the pdf.
    :type x:  2D ndarray (nsamples, dimension)
                nsamples: an integer providing the desired number of iid samples to be drawn

    :param params: list of parameters for the distribution (list of lists if distribution is defined via its
                        marginals)
    :type: params: list of lists or ndarray

    :param copula_params: Parameters of the copula
    :type: copula_params: list or ndarray

    **Output:**

    :return c: Values of the cdf
    :rtype c: ndarray
    """
    x = check_input_dims(x)
    self.update_params(params, copula_params)
    if isinstance(self.dist_name, str):
        return subdistribution_cdf(dist_name=self.dist_name, x=x, params=self.params)
    elif isinstance(self.dist_name, list):
        if (x.shape[1] != len(self.dist_name)) or (len(params) != len(self.dist_name)):
            raise ValueError('Inconsistent dimensions in inputs dist_name and params.')
        if not hasattr(self, 'copula'):
            cdfs = np.zeros_like(x)
            for i in range(len(self.dist_name)):
                cdfs[:, i] = subdistribution_cdf(dist_name=self.dist_name[i], x=x[:, i, np.newaxis], params=self.params[i])
            return np.prod(cdfs, axis=1)
        else:
            c, _ = self.copula.evaluate_copula(x=x, dist_params=params, copula_params=copula_params)
            return c


# Method that computes the icdf
def define_icdf(self, x, params=None):
    """
    A method that computes the cumulative distribution function at inputs points x. Only for univariate distributions.

    **Input:**

    :param x: Points to estimate the pdf.
    :type x:  2D ndarray (nsamples, dimension)
                nsamples: an integer providing the desired number of iid samples to be drawn

    :param params: list of parameters for the distribution (list of lists if distribution is defined via its
                        marginals)
    :type: params: list of lists or ndarray

    **Output:**

    :return c: Values of the cdf
    :rtype c: ndarray
    """
    x = check_input_dims(x)
    self.update_params(params, copula_params=None)
    if isinstance(self.dist_name, str):
        return subdistribution_icdf(dist_name=self.dist_name, x=x, params=self.params)
    elif isinstance(self.dist_name, list):
        raise AttributeError('Method icdf not defined for multivariate distributions.')


# Method that generates RVs
def define_rvs(self, nsamples=1, params=None):
    """
    A method that samples iid realizations from the distribution - does not support distributions with copula

    **Input:**

    :param nsamples: An integer providing the desired number of iid samples to be drawn.

                    Default: 1
    :type nsamples:  int

    :param params: list of parameters for the distribution (list of lists if distribution is defined via its
                        marginals)
    :type: params: list of lists or ndarray

    **Output:**

    :return rvs: Realizations from the distribution
    :rtype rvs: ndarray
    """
    self.update_params(params, copula_params=None)
    if isinstance(self.dist_name, str):
        return subdistribution_rvs(dist_name=self.dist_name, nsamples=nsamples, params=self.params)
    elif isinstance(self.dist_name, list):
        if len(self.params) != len(self.dist_name):
            raise ValueError('UQpy error: Inconsistent dimensions')
        if not hasattr(self, 'copula'):
            rvs = np.zeros((nsamples, len(self.dist_name)))
            for i in range(len(self.dist_name)):
                rvs[:, i] = subdistribution_rvs(dist_name=self.dist_name[i], nsamples=nsamples, params=self.params[i])[:, 0]
            return rvs
        else:
            raise AttributeError('Method rvs not defined for distributions with copula.')


def define_fit(self, x):
    """
    Compute the MLE parameters of a distribution from data x - does not support distributions with copula

    **Input:**

    :param x: Vector of data x
    :type x:  ndarray

    **Output:**

    :return params_fit: MLE parameters.
    :rtype params_fit: ndarray
    """
    x = check_input_dims(x)
    if isinstance(self.dist_name, str):
        return subdistribution_fit(dist_name=self.dist_name, x=x)
    elif isinstance(self.dist_name, list):
        if x.shape[1] != len(self.dist_name):
            raise ValueError('Inconsistent dimensions in inputs dist_name and x.')
        if not hasattr(self, 'copula'):
            params_fit = []
            for i in range(len(self.dist_name)):
                params_fit.append(subdistribution_fit(dist_name=self.dist_name[i], x=x[:, i, np.newaxis]))
            return params_fit
        else:
            raise AttributeError('Method fit not defined for distributions with copula.')


# Method that computes moments
def define_moments(self, params=None):
    """
    Compute marginal moments (mean, variance, skewness, kurtosis) - does not support distributions with copula.

    **Input:**

    :param params: list of parameters for the distribution (list of lists if distribution is defined via its
                        marginals)
    :type: params: list of lists or ndarray

    **Output:**

    :return mean: Mean value.
    :rtype mean: list

    :return var: Variance.
    :rtype var: list

    :return skew: Kyrtosis.
    :rtype skew: list

    :return kurt: Kyrtosis.
    :rtype kurt: list
     """
    self.update_params(params, copula_params=None)
    if isinstance(self.dist_name, str):
        return subdistribution_moments(dist_name=self.dist_name, params=self.params)
    elif isinstance(self.dist_name, list):
        if len(self.params) != len(self.dist_name):
            raise ValueError('UQpy error: Inconsistent dimensions')
        if not hasattr(self, 'copula'):
            mean, var, skew, kurt = [0]*len(self.dist_name), [0]*len(self.dist_name), [0]*len(self.dist_name), \
                                    [0]*len(self.dist_name),
            for i in range(len(self.dist_name)):
                mean[i], var[i], skew[i], kurt[i] = subdistribution_moments(dist_name=self.dist_name[i],
                                                                            params=self.params[i])
            return mean, var, skew, kurt
        else:
            raise AttributeError('Method moments not defined for distributions with copula.')


class Copula:
    """
    This class computes terms required to compute cdf, pdf and log_pdf for a multivariate distribution whose
    dependence structure is defined with a copula. The following copula are supported:  Gumbel

    **Input:**
    :param copula_name: Name of copula.
    :type: copula_name: string

    :param dist_name: names of the marginal distributions.
    :type: dist_name: list of strings

    **Output:**
    A handler pointing to a copula and its associated methods, in particular its method evaluate_copula, which
     evaluates the terms c, c_ necessary to evaluate the cdf and pdf, respectively, of the multivariate
    Distribution.
    """

    def __init__(self, copula_name=None, dist_name=None):

        if copula_name is None or dist_name is None:
            raise ValueError('Both copula_name and dist_name must be provided.')
        self.copula_name = copula_name
        self.dist_name = dist_name

    def evaluate_copula(self, x, dist_params, copula_params):
        """ Computes the copula cdf c and copula density c_ """
        if self.copula_name.lower() == 'gumbel':
            if x.shape[1] > 2:
                raise ValueError('Maximum dimension for the Gumbel Copula is 2.')
            if not isinstance(copula_params, list):
                copula_params = [copula_params]
            if copula_params[0] < 1:
                raise ValueError('The parameter for Gumbel copula must be defined in [1, +oo)')

            uu = np.zeros_like(x)
            for i in range(uu.shape[1]):
                uu[:, i] = subdistribution_cdf(dist_name=self.dist_name[i], x=x[:, i, np.newaxis], params=dist_params[i])
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


# Helper function: does the method exist?
def exist_method(method, dist_name, has_copula):
    """ This function returns True if this method should exist for that particular Distribution, it depends on the
    method itself, dist_name and the existence of a copula"""
    if isinstance(dist_name, str):    # all scipy supported distributions have a pdf method
        return subdistribution_exist_method(dist_name=dist_name, method=method)
    elif isinstance(dist_name, (list, tuple)):    # Check that all non-scipy
        if method == 'icdf':
            return False
        if has_copula and (method in ['moments', 'fit', 'rvs']):
            return False
        if all([subdistribution_exist_method(dist_name=n, method=method) for n in dist_name]):
            return True
    return False


# Helper functions for subdistributions, i.e., distributions where dist_name is only a string
def subdistribution_exist_method(dist_name, method):
    # Univariate scipy distributions have all methods
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
    # If it is a supported scipy distribution:
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


def subdistribution_log_pdf(dist_name, x, params):
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


def subdistribution_cdf(dist_name, x, params):
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


def subdistribution_rvs(dist_name, nsamples, params):
    # If it is a supported scipy distribution:
    if dist_name.lower() in (list_univariates + list_multivariates):
        d, kwargs = scipy_distributions(dist_name=dist_name, params=params)
        rvs = d.rvs(size=nsamples, **kwargs)
    # Otherwise it must be a file
    else:
        custom_dist = importlib.import_module(dist_name)
        tmp = getattr(custom_dist, 'rvs')
        rvs = tmp(nsamples=nsamples, params=params)
    if isinstance(rvs, (float, int)):
        return np.array([[rvs]])    # one sample in a 1d space
    if len(rvs.shape) == 1:
        if nsamples == 1:
            return rvs[np.newaxis, :]    # one sample in a d-dimensional space
        else:
            return rvs[:, np.newaxis]    # several samples in a one-dimensional space
    return rvs


def subdistribution_fit(dist_name, x):
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


def subdistribution_icdf(dist_name, x, params):
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


def scipy_distributions(dist_name, params=None):
    """ This function returns the scipy distribution, frozen with parameters in place if they are provided """
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
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

"""This module contains functionality for all the transformations supported in UQpy."""

from UQpy.Utilities import *
from UQpy.Distributions import *
from scipy.linalg import cholesky


class Dependence:
    """
    Description:
    A class to correlate or decorrelate standard normal samples ~ N(0, 1) given a correlation matrix.
    Input:
    :param action: The action to be performed (Correlate or decorellate the samples).
    :type input_samples: string
    :param input_samples: An object of SampleMethods or Dependence class or an array of standard normal samples ~ N(0, 1).
    :type input_samples: object or ndarray
    :param corr_norm: The correlation matrix of the random variables in the standard normal space.
    :type corr_norm: ndarray
    param: distribution: An object list containing the distributions of the random variables.
                         Each item in the list is an object of the Distribution class (see Distributions.py).
                         The list has length equal to dimension.
    :type distribution: list
    Output:
    :return: Dependence.samples: Set of correlated or uncorrelated normal samples.
    :rtype: Dependence.samples: ndarray

    """

    # Authors: Dimitris G.Giovanis
    # Last Modified: 10/28/19 by Dimitris G. Giovanis

    def __init__(self, input_samples=None, corr_norm=None):
        self.flag = 0  # a flag to handle the case where the correlation matrix is the identity matrix
        self.corr_norm = corr_norm

        """ If the input samples are of type object or array"""
        if not isinstance(input_samples, np.ndarray):
            _dict = {**input_samples.__dict__}
            for k, v in _dict.items():
                setattr(self, k, v)
            for i in range(len(self.dist_name)):
                if self.dist_name[i].lower() != 'normal' or self.dist_params[i] != [0, 1]:
                    raise RuntimeError("In order to use class 'Dependence' or the random variables should be standard"
                                       "normal")
        elif isinstance(input_samples, np.ndarray) is True:
            self.dist_name = ['normal'] * input_samples.shape[1]
            self.dist_params = [[0, 1]] * input_samples.shape[1]
            self.samples = input_samples

        if self.corr_norm is None:
            raise RuntimeError("A correlation matrix is required.")
        elif self.corr_norm is not None:
            self.dimension = self.corr_norm.shape[0]

        self.distribution = [None] * self.dimension
        for i in range(self.dimension):
            self.distribution[i] = Distribution(self.dist_name[i])

        if np.linalg.norm(self.corr_norm - np.identity(n=self.corr_norm.shape[0])) < 10 ** (-8):
            print("The provided correlation matrix is the identity. No action will be performed")
            self.flag = 1

    def correlate(self):
        c = cholesky(self.corr_norm, lower=True)
        if self.flag == 0:
            self.samples = np.dot(c, self.samples.copy().T).T
        if self.flag == 1:
            self.samples = self.samples.copy()

    def decorrelate(self):
        c = cholesky(self.corr_norm, lower=True)
        if self.flag == 0:
            self.samples = np.dot(np.linalg.inv(c), self.samples.copy().T).T
        if self.flag == 1:
            self.samples = self.samples.copy()


class Nataf:

    """
        Description:
            A class to perform the Nataf transformation of samples from a user-defined distribution to N(0, 1).
        Input:
            :param input_samples: An object of type SampleMethods, ndarray and InvNataf
            :type input_samples: object

            :param dist_name: A list containing the names of the distributions of the random variables.
                            Distribution names must match those in the Distributions module.
                            If the distribution does not match one from the Distributions module, the user must provide
                            custom_dist.py.
                            The length of the string must be 1 (if all distributions are the same) or equal to dimension
            :type dist_name: string list

            :param dist_params: Parameters of the distribution
                    Parameters for each random variable are defined as ndarrays
                    Each item in the list, dist_params[i], specifies the parameters for the corresponding distribution,
                        dist[i]
            :type dist_params: list

            param: distribution: An object list containing the distributions of the random variables.
                   Each item in the list is an object of the Distribution class (see Distributions.py).
                   The list has length equal to dimension.
            :type distribution: list

            :param corr The correlation matrix of the random variables in the parameter space.
            :type corr: ndarray

            :param itam_error1:
            :type itam_error1: float

            :param itam_error2:
            :type itam_error1: float

            :param beta:
            :type itam_error1: float
        Output:
            :return: Nataf.corr: The distorted correlation matrix of the random variables in the standard space;
            :rtype: Nataf.corr: ndarray

            :return: Nataf.samplesN01: An array of N(0,1) samples;
            :rtype: Nataf.corr: ndarray

            :return: Nataf.samples: An array of samples following the normal distribution.
            :rtype: Nataf.corr: ndarray

            :return: Nataf.jacobian: An array containing the Jacobian of the transformation.
            :rtype: Nataf.jacobian: ndarray
    """
    # Authors: Dimitris G. Giovanis
    # Last Modified: 10/28/2019 by Dimitris G. Giovanis

    def __init__(self, input_samples=None, corr_norm=None, corr=None, dist_name=None, dist_params=None,
                 dimension=None, beta=None, itam_error1=None, itam_error2=None):

        self.beta = beta
        self.itam_error1 = itam_error1
        self.itam_error2 = itam_error2

        if input_samples is not None:

            if isinstance(input_samples, np.ndarray):
                self.input_samples = input_samples
                self.dist_name = dist_name
                self.dist_params = dist_params
                self.corr_norm = corr_norm
                self.corr = corr
                self.dimension = dimension

            else:
                _dict = {**input_samples.__dict__}
                for k, v in _dict.items():
                    setattr(self, k, v)
                if not hasattr(self, 'corr_norm'):
                    self.corr_norm = corr_norm
                if not hasattr(self, 'corr'):
                    self.corr = corr
                self.input_samples = self.samples.copy()

                # Check whether the input samples are all standard normal:
                count = 0
                for i in range(len(self.dist_name)):
                    if self.dist_name[i].lower() == 'normal' and self.dist_params[i] == [0, 1]:
                        count = count + 1

                if count == len(self.dist_name):
                    self.dist_name = dist_name
                    self.dist_params = dist_params

        if input_samples is None:
            self.input_samples = input_samples
            self.dist_name = dist_name
            self.dist_params = dist_params
            self.corr_norm = corr_norm
            self.corr = corr
            self.dimension = dimension

        if self.dist_name is None or self.dist_params is None:
            raise RuntimeError("In order to use class 'Nataf' the distributions and their parameters must"
                               "be provided.")
        if dimension is None:
            self.dimension = len(self.dist_name)
        else:
            self.dimension = dimension

        # Ensure the dimensions of dist_name are consistent
        if type(self.dist_name).__name__ != 'list':
            self.dist_name = [self.dist_name]
        if len(self.dist_name) == 1 and self.dimension != 1:
            self.dist_name = self.dist_name * self.dimension

        # Ensure that dist_params is a list of length dimension
        if type(self.dist_params).__name__ != 'list':
            self.dist_params = [self.dist_params]
        if len(self.dist_params) == 1 and self.dimension != 1:
            self.dist_params = self.dist_params * self.dimension

        self.distribution = [None] * self.dimension
        for j in range(len(self.dist_name)):
            self.distribution[j] = Distribution(self.dist_name[j])

    def inverse(self):
        samples_n01 = self.input_samples.copy()
        if self.corr_norm is None:
            self.corr_norm = np.eye(self.dimension)
        self.samples, self.jacobian = transform_g_to_ng(self.corr_norm, self.distribution, self.dist_params,
                                                        samples_n01)

    def transform(self):
        samples_ng = self.input_samples.copy()
        if self.corr is None:
            self.corr = np.eye(self.dimension)
        self.samples, self.jacobian = transform_ng_to_g(self.corr, self.distribution, self.dist_params,
                                                        samples_ng)

    def correlation_distortion_inv(self):
        if self.corr_norm is not None:
            ident_ = np.linalg.norm(self.corr_norm - np.identity(n=self.dimension)) > 10 ** (-8)
            if ident_:
                self.corr = correlation_distortion(self.distribution, self.dist_params, self.corr_norm)
            else:
                self.corr = self.corr_norm.copy()
        else:
            raise RuntimeError("In order to calculate the correlation distortion"
                               "a correlation matrix is required.")

    def correlation_distortion(self):
        if self.corr is not None:
            i_check_ = np.linalg.norm(self.corr - np.identity(n=self.dimension)) > 10 ** (-8)
            if i_check_:
                self.corr_norm = itam(self.distribution, self.dist_params, self.corr, self.beta, self.itam_error1,
                                      self.itam_error2)
            else:
                self.corr_norm = self.corr.copy()
        else:
            raise RuntimeError("In order to calculate the correlation distortion"
                               "a correlation matrix is required.")

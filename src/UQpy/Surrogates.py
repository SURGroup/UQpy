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

"""This module contains functionality for all the surrogate methods supported in UQpy."""

import numpy as np
from UQpy.Distributions import *


########################################################################################################################
########################################################################################################################
#                                         Stochastic Reduced Order Model  (SROM)                                       #
########################################################################################################################
########################################################################################################################

class SROM:

    """
    Stochastic Reduced Order Model(SROM) provide a low-dimensional, discrete approximation of a given random
    quantity.
    SROM generates a discrete approximation of continuous random variables. The probabilities/weights are
    considered to be the parameters for the SROM and they can be obtained by minimizing the error between the
    marginal distributions, first and second order moments about origin and correlation between random variables.
    References:
    M. Grigoriu, "Reduced order models for random functions. Application to stochastic problems",
        Applied Mathematical Modelling, Volume 33, Issue 1, Pages 161-175, 2009.
    Input:
    :param samples: An array/list of samples corresponding to each random variables

    :param cdf_target: A list of Cumulative distribution functions of random variables
    :type cdf_target: list str or list function

    :param cdf_target_params: Parameters of distribution
    :type cdf_target_params: list

    :param moments: A list containing first and second order moment about origin of all random variables

    :param weights_errors: Weights associated with error in distribution, moments and correlation.
                           Default: weights_errors = [1, 0.2, 0]
    :type weights_errors: list

    :param properties: A list of booleans representing properties, which are required to match in reduce
                       order model. This class focus on reducing errors in distribution, first order moment
                       about origin, second order moment about origin and correlation of samples.
                       Default: properties = [True, True, True, False]
                       Example: properties = [True, True, False, False] will minimize errors in distribution and
                       errors in first order moment about origin in reduce order model.
    :type properties: list

    :param weights_distribution: An list or array containing weights associated with different samples.
                                 Options:
                                    If weights_distribution is None, then default value is assigned.
                                    If size of weights_distribution is 1xd, then it is assigned as dot product
                                        of weights_distribution and default value.
                                    Otherwise size of weights_distribution should be equal to Nxd.
                                 Default: weights_distribution = Nxd dimensional array with all elements equal
                                 to 1.

    :param weights_moments: An array of dimension 2xd, where 'd' is number of random variables. It contain
                            weights associated with moments.
                            Options:
                                If weights_moments is None, then default value is assigned.
                                If size of weights_moments is 1xd, then it is assigned as dot product
                                    of weights_moments and default value.
                                Otherwise size of weights_distribution should be equal to 2xd.
                            Default: weights_moments = Square of reciprocal of elements of moments.
    :type weights_moments: ndarray or list (float)

    :param weights_correlation: An array of dimension dxd, where 'd' is number of random variables. It contain
                                weights associated with correlation of random variables.
                                Default: weights_correlation = dxd dimensional array with all elements equal to
                                1.

    :param correlation: Correlation matrix between random variables.

    Output:
    :return: SROM.sample_weights: The probabilities weights for each sample as identified through optimization.
    :rtype: SROM.sample_weights: ndarray
    """
    # Authors: Mohit Chauhan
    # Updated: 6/7/18 by Dimitris G. Giovanis

    def __init__(self, samples=None, cdf_target=None, moments=None, weights_errors=None,
                 weights_distribution=None, weights_moments=None, weights_correlation=None,
                 properties=None, cdf_target_params=None, correlation=None):

        if type(weights_distribution) is list:
            self.weights_distribution = np.array(weights_distribution)
        else:
            self.weights_distribution = weights_distribution

        if type(weights_moments) is list:
            self.weights_moments = np.array(weights_moments)
        else:
            self.weights_moments = weights_moments

        if type(correlation) is list:
            self.correlation = np.array(correlation)
        else:
            self.correlation = correlation

        if type(moments) is list:
            self.moments = np.array(moments)
        else:
            self.moments = moments
        if type(samples) is list:
            self.samples = np.array(samples)
            self.nsamples = self.samples.shape[0]
            self.dimension = self.samples.shape[1]
        else:
            self.dimension = samples.shape[1]
            self.samples = samples
            self.nsamples = samples.shape[0]

        if type(weights_correlation) is list:
            self.weights_correlation = np.array(weights_correlation)
        else:
            self.weights_correlation = weights_correlation

        self.weights_errors = weights_errors
        self.cdf_target = cdf_target
        self.properties = properties
        self.cdf_target_params = cdf_target_params
        self.init_srom()
        self.sample_weights = self.run_srom()

    def run_srom(self):
        print('UQpy: Performing SROM...')
        from scipy import optimize

        def f(p0, samples, wd, wm, wc, mar, n, d, m, alpha, para, prop, correlation):
            e1 = 0.
            e2 = 0.
            e22 = 0.
            e3 = 0.
            com = np.append(samples, np.transpose(np.matrix(p0)), 1)
            for j in range(d):
                srt = com[np.argsort(com[:, j].flatten())]
                s = srt[0, :, j]
                a = srt[0, :, d]
                a0 = np.cumsum(a)
                marginal = mar[j]

                if prop[0] is True:
                    for i in range(n):
                        e1 += wd[i, j] * (a0[0, i] - marginal(s[0, i], para[j])) ** 2

                if prop[1] is True:
                    e2 += wm[0, j] * (np.sum(np.array(p0) * samples[:, j]) - m[0, j]) ** 2

                if prop[2] is True:
                    e22 += wm[1, j] * (
                            np.sum(np.array(p0) * (samples[:, j] * samples[:, j])) - m[1, j]) ** 2

                if prop[3] is True:
                    for k in range(d):
                        if k > j:
                            r = correlation[j, k] * np.sqrt((m[1, j] - m[0, j] ** 2) * (m[1, k] - m[0, k] ** 2)) + \
                                m[0, j] * m[0, k]
                            e3 += wc[k, j] * (
                                    np.sum(np.array(p_) * (
                                                np.array(samples[:, j]) * np.array(samples[:, k]))) - r) ** 2

            return alpha[0] * e1 + alpha[1] * (e2 + e22) + alpha[2] * e3

        def constraint(x):
            return np.sum(x) - 1

        def constraint2(y):
            n = np.size(y)
            return np.ones(n) - y

        def constraint3(z):
            n = np.size(z)
            return z - np.zeros(n)

        cons = ({'type': 'eq', 'fun': constraint}, {'type': 'ineq', 'fun': constraint2},
                {'type': 'ineq', 'fun': constraint3})

        p_ = optimize.minimize(f, np.zeros(self.nsamples),
                               args=(self.samples, self.weights_distribution, self.weights_moments,
                                     self.weights_correlation, self.cdf_target, self.nsamples, self.dimension,
                                     self.moments, self.weights_errors, self.cdf_target_params, self.properties,
                                     self.correlation),
                               constraints=cons, method='SLSQP')

        print('Done!')
        return p_.x

    def init_srom(self):

        if self.cdf_target is None:
            raise NotImplementedError("Exit code: Distribution not defined.")

        # Check samples
        if self.samples is None:
            raise NotImplementedError('Samples not provided for SROM')

        # Check properties to match
        if self.properties is None:
            self.properties = [True, True, True, False]

        # Check moments and correlation
        if self.properties[1] is True or self.properties[2] is True or self.properties[3] is True:
            if self.moments is None:
                raise NotImplementedError("'moments' are required")
        # Both moments are required, if correlation property is required to be match
        if self.properties[3] is True:
            if self.moments.shape != (2, self.dimension):
                raise NotImplementedError("1. Size of 'moments' is not correct")
            if self.correlation is None:
                self.correlation = np.identity(self.dimension)
        # moments.shape[0] should be 1 or 2
        if self.moments.shape != (1, self.dimension) and self.moments.shape != (2, self.dimension):
            raise NotImplementedError("2. Size of 'moments' is not correct")
        # If both the moments are to be included in objective function, then moments.shape[0] should be 2
        if self.properties[1] is True and self.properties[2] is True:
            if self.moments.shape != (2, self.dimension):
                raise NotImplementedError("3. Size of 'moments' is not correct")
        # If only second order moment is to be included in objective function and moments.shape[0] is 1. Then
        # self.moments is converted shape = (2, self.dimension) where is second row contain second order moments.
        if self.properties[1] is False and self.properties[2] is True:
            if self.moments.shape == (1, self.dimension):
                temp = np.ones(shape=(1, self.dimension))
                self.moments = np.concatenate((temp, self.moments))

        # Check weights corresponding to errors
        if self.weights_errors is None:
            self.weights_errors = [1, 0.2, 0]
        self.weights_errors = np.array(self.weights_errors).astype(np.float64)

        # Check weights corresponding to distribution
        if self.weights_distribution is None:
            self.weights_distribution = np.ones(shape=(self.samples.shape[0], self.dimension))

        self.weights_distribution = np.array(self.weights_distribution)
        if self.weights_distribution.shape == (1, self.dimension):
            self.weights_distribution = self.weights_distribution * np.ones(shape=(self.samples.shape[0],
                                                                                   self.dimension))
        elif self.weights_distribution.shape != (self.samples.shape[0], self.dimension):
            raise NotImplementedError("Size of 'weights for distribution' is not correct")

        # Check weights corresponding to moments and it's default list
        if self.weights_moments is None:
            self.weights_moments = np.reciprocal(np.square(self.moments))

        self.weights_moments = np.array(self.weights_moments)
        if self.weights_moments.shape == (1, self.dimension):
            self.weights_moments = self.weights_moments * np.ones(shape=(2, self.dimension))
        elif self.weights_moments.shape != (2, self.dimension):
            raise NotImplementedError("Size of 'weights for moments' is not correct")

        # Check weights corresponding to correlation and it's default list
        if self.weights_correlation is None:
            self.weights_correlation = np.ones(shape=(self.dimension, self.dimension))

        self.weights_correlation = np.array(self.weights_correlation)
        if self.weights_correlation.shape != (self.dimension, self.dimension):
            raise NotImplementedError("Size of 'weights for correlation' is not correct")

        # Check cdf_target
        if len(self.cdf_target) == 1:
            self.cdf_target = self.cdf_target * self.dimension
            self.cdf_target_params = [self.cdf_target_params] * self.dimension
        elif len(self.cdf_target) != self.dimension:
            raise NotImplementedError("Size of cdf_type should be 1 or equal to dimension")

        # Assign cdf_target function for each dimension
        for i in range(len(self.cdf_target)):
            if type(self.cdf_target[i]).__name__ == 'function':
                self.cdf_target[i] = self.cdf_target[i]
            elif type(self.cdf_target[i]).__name__ == 'str':
                self.cdf_target[i] = cdf(self.cdf_target[i])
            else:
                raise NotImplementedError("Distribution type should be either 'function' or 'list'")

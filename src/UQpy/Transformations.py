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

# Authors: Dimitris G.Giovanis
# Last Modified: 1/2/2020 by D G. Giovanis.


class Nataf:

    """
        Description:
        A class to perform iso-probabilistic transformations of random variables between standard normal space and
        physical space
        """

    # Authors: Dimitris G.Giovanis
    # Last Modified: 10/28/19 by Dimitris G. Giovanis

    def __init__(self, corr=None, dist_name=None, dist_params=None, dimension=None, beta=None,
                 itam_error1=None, itam_error2=None):

        self.beta = beta
        self.itam_error1 = itam_error1
        self.itam_error2 = itam_error2
        self.dist_name = dist_name
        self.corr = corr
        self.dimension = dimension
        self.dist_params = dist_params
        self.distribution = list()
        for j in range(len(self.dist_name)):
            self.distribution.append(Distribution(self.dist_name[j]))

    def transform(self, samples):
        self.corr_z = self.distortion_x_to_z(self.distribution, self.dist_params, self.corr, self.beta, self.itam_error1
                                             , self.itam_error2)
        self.u, self.jacobian_x_to_u = self.transform_x_to_u(samples, self.corr_z, self.distribution,
                                                             self.dist_params, jacobian=True)

    def inverse(self, samples):
        self.corr_x = self.distortion_z_to_x(self.distribution, self.dist_params, self.corr)

        self.x, self.jacobian_u_to_x = self.transform_u_to_x(samples, self.corr, self.distribution,
                                                             self.dist_params, jacobian=True)

    @staticmethod
    def distortion_x_to_z(distribution, dist_params, corr_x, beta=None, itam_error1=None, itam_error2=None):
        """For estimating the correlation in the normal space z given the correlation in the x space"""
        corr_z = itam(distribution, dist_params, corr_x, beta, itam_error1, itam_error2)
        return corr_z

    @staticmethod
    def distortion_z_to_x(distribution, dist_params, corr_z):
        """For estimating the correlation in the physical space x given the correlation in the normal z space"""
        corr_x = correlation_distortion(distribution, dist_params, corr_z)
        return corr_x

    @staticmethod
    def transform_x_to_z(x, dist, dist_params):
        """
            Description:
                Perform the transformation between original space x and  normal space z for a random variable
                with given probability distribution.
            Input:
                :param x: sample in physical space
                :type x: array/float
                :param dist: marginal distributions
                :type dist: list
                :param dist_params: marginal distribution parameters
                :type dist_params: list
                :param x: non-Gaussian samples
                :type x: array
            Output:
                :return: z: Gaussian samples
                :rtype: z: array
        """
        z = np.zeros_like(x)
        m, n = np.shape(x)
        for j in range(n):
            cdf = dist[j].cdf
            z[:, j] = stats.norm.ppf(cdf(x[:, j][:, np.newaxis], dist_params[j]))
        return z

    @staticmethod
    def transform_z_to_x(z, dist, dist_params):
        """
            Description:
                Perform the transformation between original space x and  normal space z for a random variable
                with given probability distribution.
            Input:
                :param z: sample
                :type z: array/float
                :param dist: marginal distributions
                :type dist: list
                :param dist_params: marginal distribution parameters
                :type dist_params: list
            Output:
                :return: x: Gaussian samples
                :rtype: x: array
        """
        x = np.zeros_like(z)
        m, n = np.shape(x)
        for j in range(n):
            i_cdf = dist[j].icdf
            x[:, j] = i_cdf(stats.norm.cdf(z[:, j][:, np.newaxis]), dist_params[j])
        return x

    @staticmethod
    def transform_u_to_z(u, corr_norm):

        """
            Description:
                Perform the transformation between standard normal space and correlated normal space.
            Input:
                :param corr_norm: Correlation matrix in the standard normal space
                :type corr_norm: array
                :param u: Gaussian samples
                :type u: array

            Output:
                :return: z: Gaussian samples
                :rtype: z: array

        """

        from scipy.linalg import cholesky
        l0 = cholesky(corr_norm, lower=True)
        z = np.dot(l0, u.T).T
        return z

    @staticmethod
    def transform_z_to_u(z, corr_norm):

        """
            Description:
                Perform the transformation between correlated normal space and standard normal space.
            Input:
                :param corr_norm: Correlation matrix in the standard normal space
                :type corr_norm: array
                :param z: Gaussian samples
                :type z: array
            Output:
                :return: u: Gaussian samples
                :rtype: u: array

        """

        from scipy.linalg import cholesky
        l0 = cholesky(corr_norm, lower=True)
        u = np.dot(np.linalg.inv(l0), z.T).T
        return u

    @staticmethod
    def transform_x_to_u(x, corr_norm, dist, dist_params, jacobian=True):
        """
            Description:
                Perform the transformation between original space x and standard normal space u for a random variable
                with given probability distribution.
            Input:
                :param corr_norm: Correlation matrix in the standard normal space
                :type corr_norm: array
                :param dist: marginal distributions
                :type dist: list
                :param dist_params: marginal distribution parameters
                :type dist_params: list
                :param x: non-Gaussian samples
                :type x: array
                :param jacobian: The Jacobian of the transformation
                :type jacobian: array
            Output:
                :return: samples_g: Gaussian samples
                :rtype: samples_g: ndarray

                :return: jacobian: The jacobian
                :rtype: jacobian: ndarray

        """
        from scipy.linalg import cholesky
        l0 = cholesky(corr_norm, lower=True)
        z = Nataf.transform_x_to_z(x, dist, dist_params)
        u = np.dot(np.linalg.inv(l0), z.T).T

        if not jacobian:
            return u
        else:
            m, n = np.shape(u)
            y = np.zeros(shape=(n, n))
            jacobian_x_to_u = [None] * m
            for i in range(m):
                for j in range(n):
                    pdf = dist[j].pdf
                    x0 = np.array([x[i, j]])
                    xi = np.array([u[i, j]])
                    y[j, j] = stats.norm.pdf(xi[:, np.newaxis]) / pdf(x0[:, np.newaxis], dist_params[j])
                jacobian_x_to_u[i] = np.linalg.solve(y, l0)

            return u, jacobian_x_to_u

    @staticmethod
    def transform_u_to_x(u, corr_norm, dist, dist_params, jacobian=True):
        """
            Description:
                perform the transformation between standard normal space u and original space x.
            Input:
                :param corr_norm: Correlation matrix in the standard normal space
                :type corr_norm: array
                :param dist: marginal distributions
                :type dist: list
                :param dist_params: marginal distribution parameters
                :type dist_params: list
                :param u: Gaussian samples
                :type u: array
                :param jacobian: The Jacobian of the transformation
                :type jacobian: array
            Output:
                :return: x: Gaussian samples
                :rtype: x: array

                :return: jacobian: The jacobian
                :rtype: jacobian: array

        """

        from scipy.linalg import cholesky
        l0 = cholesky(corr_norm, lower=True)
        z = np.dot(l0, u.T).T
        m, n = np.shape(u)
        x = Nataf.transform_z_to_x(z, dist, dist_params)

        if not jacobian:
            return x, None
        else:
            temp_ = np.zeros([n, n])
            jacobian_u_to_x = [None] * m
            for i in range(m):
                for j in range(n):
                    pdf = dist[j].pdf
                    xi = np.array([x[i, j]])
                    x0 = np.array([z[i, j]])
                    temp_[j, j] = pdf(xi[:, np.newaxis], dist_params[j]) / stats.norm.pdf(x0[:, np.newaxis])
                jacobian_u_to_x[i] = np.linalg.solve(l0, temp_)

            return x, jacobian_u_to_x


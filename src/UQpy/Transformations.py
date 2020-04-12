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

"""This module contains functionality for all the transformations supported in UQpy.The module currently contains the
following classes:
* Nataf: Define the Nataf transformation and its inverse between distributions.
"""

from UQpy.Utilities import *
from UQpy.Distributions import *


class Nataf:
    """
    Perform the Nataf transformation and its inverse.

    This class performs the  iso-probabilistic transformations, most notably the Nataf transformation
    to transform arbitrarily distributed random variables to standard normal variables.

    **References:**

    1. A. Nataf, “Determination des distributions dont les marges sont donnees”, C. R. Acad. Sci.
       vol. 225, pp. 42-43, Paris, 1962.
    2. R. Lebrun and A. Dutfoy, “An innovating analysis of the Nataf transformation from the copula viewpoint”,
       Prob. Eng. Mech.,  vol. 24, pp. 312-320, 2009.

    **Input:**

    :param corr: The covariance matrix in case of correlated variables.
                    Default: None
    :type corr: ndarray

    :param dist_name: A list containing the names of the distributions of the random variables.
                      Distribution names must match those in the Distributions module.
                      If the distribution does not match one from the Distributions module, the user must
                      provide custom_dist.py. The length of the string must be 1 (if all distributions are the same)
                      or equal to dimension.
    :type dist_name: string list

    :param dist_params: Parameters of the distribution.
    :type dist_params: ndarray or list

    :param dimension: Number of random variables.

                      This object must be specified.

                      Default: None
    :type dimension: int

    :param beta: A variable selected to optimize convergence speed and desired accuracy of the ITAM method.

                 Default: 1
    :type beta: int

    :param itam_error1: A threshold value the ITAM method.

                        Default: 0.0001
    :type itam_error1: float

    :param itam_error2: A threshold value the ITAM method.

                        Default: 0.01
    :type itam_error2: float

    **Attributes:**

    :param self.corr_z: The correlation matrix in the normal space z.
    :type self.corr_z: ndarray

    :param self.corr_x: The correlation matrix in the physical space x.
    :type self.corr_x: ndarray

    :param self.u: Vector of standard normal variables.
    :type self.u: ndarray

    :param self.x: Vector of random variables.
    :type self.x: ndarray

    :param self.jacobian_x_to_u: The Jacobian of the Nataf transformation.
    :type self.jacobian_x_to_u: ndarray

    :param self.jacobian_u_to_x: The Jacobian of the inverse Nataf transformation.
    :type self.jacobian_u_to_x: ndarray

    **Author:**

    Authors: Dimitris G. Giovanis

    Last Modified: 1/2/2020 by Dimitris G. Giovanis
    """

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
        for i in range(self.dimension):
            self.distribution.append(Distribution(dist_name=self.dist_name[i]))

    def transform(self, samples):
        """
        Perform Nataf transformation.

        This is a method to perform the Nataf transformation. Transform arbitrarily distributed random variables
        to standard normal variables. This is a an instance of the Nataf class.

        **Input:**

        :param samples: Samples to perform the Nataf distribution.
        :type samples: ndarray

        **Output/Returns:**

        :param self.corr_z: Correlation matrix in normal space.
        :type self.corr_z: ndarray

        :param self.u: Samples in the standard normal space.
        :type self.u: ndarray

        :param self.jacobian_x_to_u: The Jacobian of the transformation.
        :type self.jacobian_x_to_u: ndarray

        **Author:**

        Authors: Dimitris G. Giovanis

        Last Modified: 1/2/2020 by Dimitris G. Giovanis
        """
        self.corr_z = self.distortion_x_to_z(self.distribution, self.dist_params, self.corr, self.beta, self.itam_error1
                                             , self.itam_error2)
        self.u, self.jacobian_x_to_u = self.transform_x_to_u(samples, self.corr_z, self.distribution,
                                                             self.dist_params, jacobian=True)

    def inverse(self, samples):
        """
        Perform the inverse Nataf transformation.

        This is a method to perform the inverse Nataf transformation. Transform standard normal vriables to
        arbitrarily distributed random variables. This is a an instance of the Nataf class.

         **Input:**

        :param samples: Samples to perform the Nataf distribution.
        :type samples: ndarray

        **Output/Returns:**

        :param self.corr_x: Correlation matrix in physical space.
        :type self.corr_x: ndarray

        :param self.x: Samples in the physical space.
        :type self.x: ndarray

        :param self.jacobian_u_to_x: The Jacobian of the transformation.
        :type self.jacobian_u_to_x: ndarray

        **Author:**

        Authors: Dimitris G. Giovanis

        Last Modified: 1/2/2020 by Dimitris G. Giovanis

        """
        self.corr_x = self.distortion_z_to_x(self.distribution, self.dist_params, self.corr)

        self.x, self.jacobian_u_to_x = self.transform_u_to_x(samples, self.corr, self.distribution,
                                                             self.dist_params, jacobian=True)

    @staticmethod
    def distortion_x_to_z(distribution, dist_params, corr_x, beta=None, itam_error1=None, itam_error2=None):
        """
        Calculate correlation distortion.

        This is a method to calculate the correlation distortion in the normal space z given
        the correlation in the x space. This is a static method, part of the Nataf class.

        **Input:**

        :param distribution: An instance of the UQpy.Distributions class
        :type distribution: list of objects

        :param dist_params: Parameters of the distribution.
        :type dist_params: ndarray or list

        :param corr_x: Correlation of variables in the physical space
        :type corr_x: ndarray

        :param beta: A variable selected to optimize convergence speed and desired accuracy of the ITAM method.

                     Default: 1
        :type beta: int

        :param itam_error1: A threshold value the ITAM method.

                            Default: 0.0001
        :type itam_error1: float

        :param itam_error2: A threshold value the ITAM method.

                            Default: 0.01
        :type itam_error2: float

        **Output/Returns:**

        :param corr_z: Correlation matrix in normal space.
        :type corr_z: ndarray

        **Author:**

        Authors: Dimitris G. Giovanis

        Last Modified: 1/2/2020 by Dimitris G. Giovanis

        """
        corr_z = itam(distribution, dist_params, corr_x, beta, itam_error1, itam_error2)
        return corr_z

    @staticmethod
    def distortion_z_to_x(distribution, dist_params, corr_z):
        """
        Calculate correlation distortion.

        This is a method to estimate  the correlation distortion  in the physical space x
        given the correlation in the normal z space. This is a static method, part of the Nataf class.

        **Input:**

        :param distribution: An instance of the UQpy.Distributions class
        :type distribution: list of objects

        :param dist_params: Parameters of the distribution.
        :type dist_params: ndarray or list

        :param corr_z: Correlation of variables in the physical space
        :type corr_z: ndarray

        **Output/Returns:**

        :param corr_x: Correlation matrix in physical space.
        :type corr_x: ndarray

        **Author:**

        Authors: Dimitris G. Giovanis

        Last Modified: 1/2/2020 by Dimitris G. Giovanis
        """
        corr_x = correlation_distortion(distribution, dist_params, corr_z)
        return corr_x

    @staticmethod
    def transform_x_to_z(x, distribution, dist_params):
        """
        Perform the transformation from x to z.

        This is a method to perform the transformation between original space x and  normal space z
        for a random variable with given probability distribution. This is a static method,
        part of the Nataf class.

        **Input:**

        :param x: Random variables in the physical space
        :type x: ndarray or list

        :param distribution: An instance of the UQpy.Distributions class
        :type distribution: list of objects

        :param dist_params: Parameters of the distribution.
        :type dist_params: ndarray or list

        **Output/Returns:**

        :param z: Transformed samples in the normal space
        :type z: ndarray

        **Author:**

        Authors: Dimitris G. Giovanis

        Last Modified: 1/2/2020 by Dimitris G. Giovanis

        """
        z = np.zeros_like(x)
        m, n = np.shape(x)
        for j in range(n):
            cdf = distribution[j].cdf
            z[:, j] = stats.norm.ppf(cdf(x[:, j][:, np.newaxis], dist_params[j]))
        return z

    @staticmethod
    def transform_z_to_x(z, distribution, dist_params):
        """
        Perform the transformation from z to x.

        This is a method to perform the transformation between original space x and  normal space z
        for a random variable  with given probability distribution. This is a static method, part of the Nataf class.

        **Input:**

        :param z: Random variables in the normal space
        :type z: ndarray or list

        :param distribution: An instance of the UQpy.Distributions class
        :type distribution: list of objects

        :param dist_params: Parameters of the distribution.
        :type dist_params: ndarray or list

        **Output/Returns:**

        :param x: Transformed samples in the physical space
        :type x: ndarray

        **Author:**

        Authors: Dimitris G. Giovanis

        Last Modified: 1/2/2020 by Dimitris G. Giovanis

        """
        x = np.zeros_like(z)
        m, n = np.shape(x)
        for j in range(n):
            i_cdf = distribution[j].icdf
            x[:, j] = i_cdf(stats.norm.cdf(z[:, j][:, np.newaxis]), dist_params[j])
        return x

    @staticmethod
    def transform_u_to_z(u, corr_norm):
        """
        Transform u to z.

        This  is a method to perform the transformation between standard normal space and correlated normal space.
        This is a static method, part of the Nataf class.

        **Input:**

        :param u: Random variables in the standard normal space
        :type u: ndarray or list

        :param corr_norm: The correlation matrix in the normal space
        :type corr_norm: ndarray

        **Output/Returns:**

        :param z: Transformed samples in the normal space
        :type z: ndarray

        **Author:**

        Authors: Dimitris G. Giovanis

        Last Modified: 1/2/2020 by Dimitris G. Giovanis
        """
        from scipy.linalg import cholesky
        l0 = cholesky(corr_norm, lower=True)
        z = np.dot(l0, u.T).T
        return z

    @staticmethod
    def transform_z_to_u(z, corr_norm):
        """
        Transform z to u.

        This is a method to perform the transformation between correlated normal space and standard normal space.
        This is a static method, part of the Nataf class.

        **Input:**

        :param z: Random variables in the normal space
        :type z: ndarray or list

        :param corr_norm: The correlation matrix in the normal space
        :type corr_norm: ndarray

         **Output/Returns:**

        :param u: Transformed samples in the standard normal space
        :type u: ndarray

        **Author:**

        Authors: Dimitris G. Giovanis

        Last Modified: 1/2/2020 by Dimitris G. Giovanis

        """

        from scipy.linalg import cholesky
        l0 = cholesky(corr_norm, lower=True)
        u = np.dot(np.linalg.inv(l0), z.T).T
        return u

    @staticmethod
    def transform_x_to_u(x, corr_norm, distribution, dist_params, jacobian=True):
        """
        Transform x to u.

        This is a method to perform the transformation between original space x and standard normal space u
        for a random variable with given probability distribution.
        This is a static method, part of the Nataf class.

        **Input:**

        :param x: Random variables in the physical space
        :type x: ndarray or list

        :param distribution: An instance of the UQpy.Distributions class
        :type distribution: list of objects

        :param dist_params: Parameters of the distribution.
        :type dist_params: ndarray or list

        :param corr_norm: The correlation matrix in the normal space
        :type corr_norm: ndarray

        :param jacobian: The jacobian of the transformation
        :type jacobian: boolean

        **Output/Returns:**

        :param u: Transformed samples in the standard normal space
        :type u: ndarray

        :param jacobian_x_to_u: Jacobian of the transformation
        :type jacobian_x_to_u: ndarray

        **Author:**

        Authors: Dimitris G. Giovanis

        Last Modified: 1/2/2020 by Dimitris G. Giovanis

        """
        from scipy.linalg import cholesky
        l0 = cholesky(corr_norm, lower=True)
        z = Nataf.transform_x_to_z(x, distribution, dist_params)
        u = np.dot(np.linalg.inv(l0), z.T).T

        if not jacobian:
            return u
        else:
            m, n = np.shape(u)
            y = np.zeros(shape=(n, n))
            jacobian_x_to_u = [None] * m
            for i in range(m):
                for j in range(n):
                    pdf = distribution[j].pdf
                    x0 = np.array([x[i, j]])
                    xi = np.array([u[i, j]])
                    y[j, j] = stats.norm.pdf(xi[:, np.newaxis]) / pdf(x0[:, np.newaxis], dist_params[j])
                jacobian_x_to_u[i] = np.linalg.solve(y, l0)

            return u, jacobian_x_to_u

    @staticmethod
    def transform_u_to_x(u, corr_norm, distribution, dist_params, jacobian=True):
        """
        Transform u to x.

        This is a method to perform the transformation between standard normal space u and original space x.
        This is a static method, part of the Nataf class.

        **Input:**

        :param u: Random variables in the physical space
        :type u: ndarray or list

        :param distribution: An instance of the UQpy.Distributions class
        :type distribution: list of objects

        :param dist_params: Parameters of the distribution.
        :type dist_params: ndarray or list

        :param corr_norm: The correlation matrix in the normal space
        :type corr_norm: ndarray

        :param jacobian: The jacobian of the transformation
        :type jacobian: boolean

        **Output/Returns:**

        :param x: Transformed samples in the physical space
        :type x: ndarray

        :param jacobian_u_to_x: Jacobian of the transformation
        :type jacobian_u_to_x: ndarray

        **Author:**

        Authors: Dimitris G. Giovanis

        Last Modified: 1/2/2020 by Dimitris G. Giovanis
        """

        from scipy.linalg import cholesky
        l0 = cholesky(corr_norm, lower=True)
        z = np.dot(l0, u.T).T
        m, n = np.shape(u)
        x = Nataf.transform_z_to_x(z, distribution, dist_params)

        if not jacobian:
            return x, None
        else:
            temp_ = np.zeros([n, n])
            jacobian_u_to_x = [None] * m
            for i in range(m):
                for j in range(n):
                    pdf = distribution[j].pdf
                    xi = np.array([x[i, j]])
                    x0 = np.array([z[i, j]])
                    temp_[j, j] = pdf(xi[:, np.newaxis], dist_params[j]) / stats.norm.pdf(x0[:, np.newaxis])
                jacobian_u_to_x[i] = np.linalg.solve(l0, temp_)

            return x, jacobian_u_to_x

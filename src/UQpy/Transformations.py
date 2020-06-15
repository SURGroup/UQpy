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


from UQpy.Distributions import *


class Nataf:
    """
    Transform random variables  using the isoprobabilistic transformations

    **Inputs:**

    * **dist_object** ((list of ) ``Distribution`` object(s)):
                    Probability distribution of each random variable. Must be an object of type
                    ``DistributionContinuous1D`` or ``JointInd``.

    * **corr_x** (`ndarray`):
        The correlation  matrix (:math:`\mathbf{C_X}`) of the random vector **X** .

    * **corr_z** (`ndarray`):
        The correlation  matrix (:math:`\mathbf{C_Z}`) of the standard normal random vector **Z** .

        Default: The ``identity`` matrix.

    * **itam_error1** (`float`):
        A threshold value the `ITAM` method (see ``Utilities`` module).

        Default: 0.001

    * **itam_error2** (`float`):
        A threshold value the `ITAM` method (see ``Utilities`` module).

        Default: 0.01

    * **beta** (`float`):
        A variable selected to optimize convergence speed and desired accuracy of the ITAM method (see
        ``Utilities`` module).

        Default: 1.0

    **Attributes:**

    * **corr_z** (`ndarray`):
        Distorted correlation matrix (:math:`\mathbf{C_Z}`) of the standard normal vector **Z**.

    * **corr_x** (`ndarray`):
        Distorted correlation matrix (:math:`\mathbf{C_X}`) of the random vector **X**.

    * **H** (`ndarray`):
        The lower triangular matrix resulting from the Cholesky decomposition of the correlation matrix
        :math:`\mathbf{C_Z}`.

    **Methods:**
    """

    def __init__(self, dist_object, beta=1.0, itam_error1=0.001, itam_error2=0.01, corr_z=None, corr_x=None):

        if isinstance(dist_object, list):
            self.dimension = len(dist_object)
            for i in range(len(dist_object)):
                if not isinstance(dist_object[i], (DistributionContinuous1D, JointInd)):
                    raise TypeError('UQpy: A  ``DistributionContinuous1D`` or ``JointInd`` object '
                                    'must be provided.')
        else:
            if not isinstance(dist_object, (DistributionContinuous1D, JointInd)):
                raise TypeError('UQpy: A  ``DistributionContinuous1D``  or ``JointInd`` object must be provided.')

        self.dist_object = dist_object
        self.corr_x = corr_x
        self.corr_z = corr_z

        self.beta = beta
        self.itam_error1 = itam_error1
        self.itam_error2 = itam_error2
        self.corr_x = corr_x
        self.dist_object = dist_object

        if corr_x is None and corr_z is None:
            self.corr_x = np.eye(self.dimension)
            self.corr_z = np.eye(self.dimension)
        elif corr_x is not None:
            if np.all(np.equal(self.corr_x, np.eye(self.dimension))):
                self.corr_z = self.corr_x
            else:
                self.corr_z = self.distortion_x2z(self.dist_object, self.corr_x, self.beta, self.itam_error1,
                                                  self.itam_error2)
        elif corr_z is not None:
            if np.all(np.equal(self.corr_z, np.eye(self.dimension))):
                self.corr_x = self.corr_z
            else:
                self.corr_x = self.distortion_z2x(self.dist_object, self.corr_z)

        from scipy.linalg import cholesky
        self.H = cholesky(self.corr_z, lower=True)

    @staticmethod
    def distortion_x2z(dist_object, corr_x,  beta=1.0, itam_error1=0.001, itam_error2=0.01):
        """
        This is a method to calculate the correlation matrix :math:`\mathbf{C_Z}` of the standard normal random vector
        :math:`\mathbf{z}` given the correlation matrix :math:`\mathbf{C_x}` of the random vector :math:`\mathbf{x}`
        using the `ITAM` method (see ``Utilities`` class).

        **Inputs:**

        * **dist_object** ((list of ) ``Distribution`` object(s)):
                Probability distribution of each random variable. Must be an object of type
                ``DistributionContinuous1D`` or ``JointInd``.

        * **corr_x** (`ndarray`):
            The correlation  matrix (:math:`\mathbf{C_X}`) of the random vector **X** .

            Default: The ``identity`` matrix.

        * **itam_error1** (`float`):
            A threshold value the `ITAM` method (see ``Utilities`` module).

            Default: 0.001

        * **itam_error2** (`float`):
            A threshold value the `ITAM` method (see ``Utilities`` module).

            Default: 0.01

        * **beta** (`float`):
            A variable selected to optimize convergence speed and desired accuracy of the ITAM method (see
            ``Utilities`` module).

            Default: 1.0

        **Output/Returns:**

        * **cov_distorted** (`ndarray`):
            Distorted correlation matrix (:math:`\mathbf{C_z}`) of the standard normal vector **Z**.

        """
        from UQpy.Utilities import itam_correlation
        cov_distorted = itam_correlation(dist_object, corr_x, beta, itam_error1, itam_error2)
        return cov_distorted

    @staticmethod
    def distortion_z2x(dist_object, corr_z):
        """
        This is a method to calculate the correlation matrix :math:`\mathbf{C_x}` of the random vector
        :math:`\mathbf{x}`  given the correlation matrix :math:`\mathbf{C_z}` of the standard normal random vector
        :math:`\mathbf{z}` using the `correlation_distortion` method (see ``Utilities`` class).

        This method is part of the ``Nataf`` class.

        **Inputs:**

        * **dist_object** ((list of ) ``Distribution`` object(s)):
                Probability distribution of each random variable. Must be an object of type
                ``DistributionContinuous1D`` or ``JointInd``.

        * **corr_z** (`ndarray`):
            The correlation  matrix (:math:`\mathbf{C_z}`) of the standard normal vector **Z** .

            Default: The ``identity`` matrix.

        **Output/Returns:**

        * **cov_distorted** (`ndarray`):
            Distorted correlation matrix (:math:`\mathbf{C_x}`) of the random vector **x**.

        """
        from UQpy.Utilities import correlation_distortion
        cov_distorted = correlation_distortion(dist_object, corr_z)
        return cov_distorted

    def transform_x2z(self, samples_x, jacobian=False):
        """
        This is a method to transform a vector :math:`\mathbf{x}` of  samples with marginal distributions
        :math:`f_i(x_i)` and cumulative distributions :math:`F_i(x_i)` to a vector :math:`\mathbf{z}` of standard normal
        samples  according to: :math:`Z_{i}=\Phi^{-1}(F_i(X_{i}))`, where :math:`\Phi` is the cumulative
        distribution function of a standard  normal variable.

        This method is part of the ``Nataf`` class.

        **Inputs:**

        * **samples_x** (`ndarray`):
            Random vector of shape ``(nsamples, dimension)`` with prescribed probability distributions.

        * **jacobian** ('Boolean'):
            A boolean whether to return the jacobian of the transformation.

            Default: False

        **Outputs:**

        * **samples_z** (`ndarray`):
            Standard normal random vector of shape ``(nsamples, dimension)``.

        * **jacobian_x2z** (`ndarray`):
            The jacobian of the transformation of shape ``(dimension, dimension)``.

        """

        m, n = np.shape(samples_x)
        samples_z = None

        if isinstance(self.dist_object, JointInd):
            if all(hasattr(m, 'cdf') for m in self.dist_object.marginals):
                samples_z = np.zeros_like(samples_x)
                for j in range(len(self.dist_object.marginals)):
                    samples_z[:, j] = stats.norm.ppf(self.dist_object.marginals[j].cdf(samples_x[:, j]))
        elif isinstance(self.dist_object, DistributionContinuous1D):
            samples_z = stats.norm.ppf(self.dist_object.cdf(samples_x))
        else:
            samples_z = np.zeros_like(samples_x)
            for j in range(n):
                samples_z[:, j] = stats.norm.ppf(self.dist_object[j].cdf(samples_x[:, j]))

        if not jacobian:
            return samples_z
        else:
            jac = np.zeros(shape=(n, n))
            jacobian_x2z = [None] * m
            for i in range(m):
                for j in range(n):
                    xi = np.array([samples_x[i, j]])
                    zi = np.array([samples_z[i, j]])
                    jac[j, j] = stats.norm.pdf(zi) / self.dist_object[j].pdf(xi)
                jacobian_x2z[i] = np.linalg.solve(jac, self.H)

            return samples_z, jacobian_x2z

    def transform_z2x(self, samples_z, jacobian=False):
        """
        This is a method to transform a standard normal vector :math:`\mathbf{z}` to a vector
        :math:`\mathbf{x}` of samples with marginal distributions :math:`f_i(x_i)` and cumulative distributions
        :math:`F_i(x_i)` to samples  according to: :math:`Z_{i}=\Phi^{-1}(F_i(X_{i}))`, where :math:`\Phi` is the
        cumulative distribution function of a standard  normal variable.

        This method is part of the ``Nataf`` class.

        **Inputs:**

        * **samples_z** (`ndarray`):
            Standard normal random vector of shape ``(nsamples, dimension)``

        * **jacobian** ('Boolean'):
            A boolean whether to return the jacobian of the transformation.

            Default: False

        **Outputs:**

        * **samples_x** (`ndarray`):
            Random vector of shape ``(nsamples, dimension)`` with prescribed probability distributions.

        * **jacobian_z2x** (`ndarray`):
            The jacobian of the transformation of shape ``(dimension, dimension)``.

        """

        m, n = np.shape(samples_z)
        from scipy.linalg import cholesky
        h = cholesky(self.corr_z, lower=True)
        #samples_z = (h @ samples_y.T).T

        samples_x = np.zeros_like(samples_z)
        if isinstance(self.dist_object, JointInd):
            if all(hasattr(m, 'icdf') for m in self.dist_object.marginals):
                for j in range(len(self.dist_object.marginals)):
                    samples_x[:, j] = self.dist_object.marginals[j].icdf(stats.norm.cdf(samples_z[:, j]))

        elif isinstance(self.dist_object, DistributionContinuous1D):
            samples_x = self.dist_object.icdf(stats.norm.cdf(samples_z))
        elif isinstance(self.dist_object, list):
            for j in range(samples_x.shape[1]):
                samples_x[:, j] = self.dist_object[j].icdf(stats.norm.cdf(samples_z[:, j]))

        if not jacobian:
            return samples_x
        else:
            jac = np.zeros(shape=(n, n))
            jacobian_z2x = [None] * m
            for i in range(m):
                for j in range(n):
                    xi = np.array([samples_x[i, j]])
                    zi = np.array([samples_z[i, j]])
                    jac[j, j] = self.dist_object[j].pdf(xi) / stats.norm.pdf(zi)
                jacobian_z2x[i] = np.linalg.solve(h, jac)

            return samples_x, jacobian_z2x

    def rvs(self, nsamples):
        """
        This is a method to generate realizations from the joint pdf of the random vector **X**.

        This method is part of the ''Nataf'' class.

        **Inputs:**

        * **nsamples** (`int`):
            Number of samples to generate.

        **Outputs:**

        * **samples_x** (`ndarray`):
            Random vector in the parameter space of shape ``(nsamples, dimension)``.

        """
        from scipy.linalg import cholesky
        h = cholesky(self.corr_z, lower=True)
        n = int(nsamples)
        m = np.size(self.dist_object)
        y = np.random.randn(nsamples, m)
        z = np.dot(h, y.T).T
        samples_x = np.zeros([n, m])
        for i in range(m):
            samples_x[:, i] = self.dist_object[i].icdf(stats.norm.cdf(z[:, i]))
        return samples_x


class Correlate:
    """
    A class to induce correlation to standard normal random variables.

    **Inputs:**

    * **samples_y** (`ndarray`):
            Uncorrelated  standard normal vector of shape ``(nsamples, dimension)``.

    * **corr_z** (`ndarray`):
        The correlation  matrix (:math:`\mathbf{C_Z}`) of the standard normal random vector **Z** .

    **Attributes:**

    * **samples_z** (`ndarray`):
        Correlated standard normal vector of shape ``(nsamples, dimension)``.

    * **H** (`ndarray`):
        The lower diagonal matrix resulting from the Cholesky decomposition of the correlation  matrix
        (:math:`\mathbf{C_Z}`).

    """

    def __init__(self, samples_y, corr_z):

        self.samples_y = samples_y
        self.corr_z = corr_z
        from scipy.linalg import cholesky
        self.H = cholesky(self.corr_z, lower=True)
        self.samples_z = (self.H @ samples_y.T).T


class Decorrelate:
    """
    A class to remove correlation from correlated standard normal random variables.


    **Inputs:**

    * **samples_z** (`ndarray`):
            Correlated standard normal vector of shape ``(nsamples, dimension)``.

    * **corr_z** (`ndarray`):
        The correlation  matrix (:math:`\mathbf{C_Z}`) of the standard normal random vector **Z** .

    **Attributes:**

    * **samples_y** (`ndarray`):
        Uncorrelated standard normal vector of shape ``(nsamples, dimension)``.

    * **H** (`ndarray`):
        The lower diagonal matrix resulting from the Cholesky decomposition of the correlation  matrix
        (:math:`\mathbf{C_Z}`).

    """
    def __init__(self, samples_z, corr_z):

        self.samples_z = samples_z
        self.corr_z = corr_z
        from scipy.linalg import cholesky
        self.H = cholesky(self.corr_z, lower=True)
        self.samples_y = np.linalg.solve(self.H, samples_z.T.squeeze()).T







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
    Transform random variables  using the `Nataf` transformation  and  calculate  the  correlation  distortion.
    ([1]_, [2]_).

    This is the parent class to all Nataf algorithms.

    **References:**

    .. [1] A. Nataf, “Determination des distributions dont les marges sont donnees”, C. R. Acad. Sci.
       vol. 225, pp. 42-43, Paris, 1962.
    .. [2] R. Lebrun and A. Dutfoy, “An innovating analysis of the Nataf transformation from the copula viewpoint”,
       Prob. Eng. Mech.,  vol. 24, pp. 312-320, 2009.

    """

    def __init__(self, dist_object):

        if isinstance(dist_object, list):
            for i in range(len(dist_object)):
                if not isinstance(dist_object[i], (DistributionContinuous1D, JointInd)):
                    raise TypeError('UQpy: A  ``DistributionContinuous1D`` or ``JointInd`` object must be provided.')
        else:
            if not isinstance(dist_object, (DistributionContinuous1D, JointInd)):
                raise TypeError('UQpy: A  ``DistributionContinuous1D``  or ``JointInd`` object must be provided.')


class Forward(Nataf):
    """
    A class perform the Nataf transformation, i.e. transform arbitrarily distributed random variables
    to independent standard normal variables. This is a an child class of the ``Nataf`` class.

    **Inputs:**

    * **samples** (`ndarray`):
            Random vector of shape ``(nsamples, dimension)`` with prescribed probability distribution, with
            ``(nsamples, dimension) = samples.shape``.

    * **dist_object** ((list of ) ``Distribution`` object(s)):
                    Probability distribution of each random variable. Must be an object of type
                    ``DistributionContinuous1D`` or ``JointInd``.

    * **cov** (`ndarray`):
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

    * **U** (`ndarray`):
        Independent standard normal vector of shape ``(nsamples, dimension)``.

    * **Cz** (`ndarray`):
        Distorted correlation matrix (:math:`\mathbf{C_Z}`) of the standard normal vector **Z**.

    * **Jxu** (`ndarray`):
        The jacobian of the transformation of shape ``(dimension, dimension)`` (if asked).

    **Methods:**
    """

    def __init__(self, dist_object, samples=None, cov=None, beta=1.0, itam_error1=0.001,
                 itam_error2=0.01):

        super().__init__(dist_object)

        self.beta = beta
        self.itam_error1 = itam_error1
        self.itam_error2 = itam_error2

        if samples is None:
            if cov is None:
                raise Warning('UQpy: No action is performed.')
            else:
                self.Cz = self.distortion_x_to_z(dist_object, cov, beta, itam_error1, itam_error2)

        else:
            if cov is None:
                self.C_z = np.eye(samples.shape[1])
                self.z = self.transform_x_to_z(samples, dist_object)
                self.u = self.z

                self.Jxu = np.eye(samples.shape[1])

            else:
                self.Cz = self.distortion_x_to_z(dist_object, cov, beta, itam_error1, itam_error2)
                z = self.transform_x_to_z(samples, dist_object)
                self.z = Correlate(z, self.Cz).z
                self.u = Uncorrelate(self.z, self.Cz).u

                self.Jxu = self.jacobian_x_u(dist_object, samples, self.u, self.Cz)

    @staticmethod
    def distortion_x_to_z(dist_object, cov, beta=1.0, itam_error1=0.001, itam_error2=0.01):
        """
        This is a method to calculate the correlation matrix :math:`\mathbf{C_Z}` of the standard normal random vector
        :math:`\mathbf{z}`  given the correlation matrix :math:`\mathbf{C_x}` of the random vector :math:`\mathbf{x}`
        using the `ITAM` method (see ``Utilities`` class).

        **Inputs:**

        * **dist_object** ((list of ) ``Distribution`` object(s)):
                Probability distribution of each random variable. Must be an object of type
                ``DistributionContinuous1D`` or ``JointInd``.

        * **cov** (`ndarray`):
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
        from UQpy.Utilities import itam
        cov_distorted = itam(dist_object, cov, beta, itam_error1, itam_error2)
        return cov_distorted

    @staticmethod
    def transform_x_to_z(x, dist_object):
        """
        This is a method to transform a vector :math:`\mathbf{x}` of  samples with marginal distributions
        :math:`f_i(x_i)` and cumulative distributions :math:`F_i(x_i)` to a vector :math:`\mathbf{z}` of standard normal
        samples  according to: :math:`Z_{i}=\Phi^{-1}(F_i(X_{i}))`, where :math:`\Phi` is the cumulative
        distribution function of a standard  normal variable.

        This is a static method, part of the ``Nataf`` class.

        **Inputs:**

        * **x** (`ndarray`):
            Random vector of shape ``(nsamples, dimension)`` with prescribed probability distributions.

        * **dist_object** ((list of) ``Distribution`` object(s)):
                    Probability distribution of each random variable. Must be an object of type
                    ``DistributionContinuous1D`` or ``JointInd``.
        **Outputs:**

        * **z** (`ndarray`):
            Standard normal random vector of shape ``(nsamples, dimension)``.

        """
        z = np.zeros_like(x)

        if isinstance(dist_object, JointInd):
            if all(hasattr(m, 'cdf') for m in dist_object.marginals):
                for j in range(len(dist_object.marginals)):
                    z[:, j] = stats.norm.ppf(dist_object.marginals[j].cdf(np.atleast_2d(x[:, j]).T))
        elif isinstance(dist_object, DistributionContinuous1D):
            f_i = dist_object.cdf
            z = np.atleast_2d(stats.norm.ppf(f_i(x))).T
        else:
            m, n = np.shape(x)
            for j in range(n):
                f_i = dist_object[j].cdf
                z[:, j] = stats.norm.ppf(f_i(np.atleast_2d(x[:, j]).T))
        return z

    @staticmethod
    def jacobian_x_u(dist_object, x, z, cov):
        """
        This is a method to calculate the jacobian of the transformation :math:`\mathbf{J}_{\mathbf{xu}}`.

        This is a static method, part of the ``Nataf`` class.

        **Inputs:**

        * **u** (`ndarray`):
            Standard normal vector of shape ``(nsamples, dimension)``.

        * **x** (`ndarray`):
            Random vector of shape ``(nsamples, dimension)``.

        * **dist_object** ((list of) ``Distribution`` object(s)):
                    Probability distribution of each random variable. Must be an object of type
                    ``DistributionContinuous1D`` or ``JointInd``.

        * **cov** (`ndarray`):
        The covariance  matrix of shape ``(dimension, dimension)``.

        **Outputs:**

        * **jacobian_x_to_u** (`ndarray`):
            Matrix of shape ``(dimension, dimension)``.

        """
        from scipy.linalg import cholesky
        h = cholesky(cov, lower=True)
        m, n = np.shape(z)
        y = np.zeros(shape=(n, n))
        jacobian_x_to_u = [None] * m
        for i in range(m):
            for j in range(n):
                xi = np.array([x[i, j]])
                zi = np.array([z[i, j]])
                y[j, j] = stats.norm.pdf(zi) / dist_object[j].pdf(xi)
            jacobian_x_to_u[i] = np.linalg.solve(y, h)

        return jacobian_x_to_u


class Inverse(Nataf):
    """
    A class perform the inverse Nataf transformation, i.e. transform independent standard normal variables to
    arbitrarily distributed random variables. This is a an child class of the ``Nataf`` class.

    **Inputs:**

    * **samples** (`ndarray``):
        Uncorrelated standard normal vector of shape ``(nsamples, dimension)`` with
        ``(nsamples, dimension) = samples.shape``.

    * **dist_object** ((list of) ``Distribution`` object(s)):
            Probability distribution of each random variable. Must be an object of type
                    ``DistributionContinuous1D`` or ``JointInd``.

    * **cov** (`ndarray`):
        The covariance  matrix of shape ``(dimension, dimension)`` (Optional) .

        Default: The ``identity`` matrix.

    **Output/Returns:**

    * **x** (`ndarray`):
        Independent standard normal vector of shape ``(nsamples, dimension)``.

    * **Cx** (`ndarray`):
        Distorted correlation matrix of the random vector **X** of shape ``(dimension, dimension)``.

    * **Jux** (`ndarray`):
        The jacobian of the transformation of shape ``(dimension, dimension)``.

    **Methods:**

    """

    def __init__(self, dist_object, samples=None, cov=None):

        super().__init__(dist_object)

        if samples is None:
            if cov is None:
                raise Warning('UQpy: No action is performed.')
            else:
                self.Cx = self.distortion_z_to_x(dist_object, cov)

        else:
            if cov is None:
                self.Cx = np.eye(samples.shape[1])
                self.z = samples
                self.x = self.transform_z_to_x(self.z, dist_object)

                self.Jux = np.eye(samples.shape[1])

            else:
                self.Cx = self.distortion_z_to_x(dist_object, cov)
                self.z = Correlate(samples, cov).z
                self.x = self.transform_z_to_x(samples, dist_object)

                self.Jux = self.jacobian_u_x(dist_object, self.z, self.x, cov)

    @staticmethod
    def distortion_z_to_x(dist_object, cov):
        """
        This is a method to calculate the correlation matrix :math:`\mathbf{C_x}` of the random vector
        :math:`\mathbf{x}`  given the correlation matrix :math:`\mathbf{C_z}` of the standard normal random vector
        :math:`\mathbf{z}` using the `correlation_distortion` method (see ``Utilities`` class). This is a static
        method, part of the ``Inverse`` class.

        **Inputs:**

        * **dist_object** ((list of ) ``Distribution`` object(s)):
                Probability distribution of each random variable. Must be an object of type
                ``DistributionContinuous1D`` or ``JointInd``.

        * **cov** (`ndarray`):
            The correlation  matrix (:math:`\mathbf{C_z}`) of the standard normal vector **z** .

            Default: The ``identity`` matrix.

        **Output/Returns:**

        * **cov_distorted** (`ndarray`):
            Distorted correlation matrix (:math:`\mathbf{C_x}`) of the random vector **x**.

        """
        from UQpy.Utilities import correlation_distortion
        cov_distorted = correlation_distortion(dist_object, cov)
        return cov_distorted

    @staticmethod
    def transform_z_to_x(z, dist_object):
        """
        This is a method to transform a vector :math:`\mathbf{z}` of  standard normal samples to a vector
        :math:`\mathbf{x}` of  samples with marginal distributions :math:`f_i(x_i)` and cumulative distributions
        :math:`F_i(x_i)` according to: :math:`Z_{i}=F_i^{-1}(\Phi(Z_{i}))`, where :math:`\Phi` is the cumulative
        distribution function of a standard  normal variable.

        This is a static method, part of the ``Nataf`` class.

        **Inputs:**

        * **z** (`ndarray`):
            Standard normal vector of shape ``(nsamples, dimension)``.

        * **dist_object** ((list of) ``Distribution`` object(s)):
                    Probability distribution of each random variable. Must be an object of type
                    ``DistributionContinuous1D`` or ``JointInd``.

        **Outputs:**

        * **x** (`ndarray`):
            Random vector of shape ``(nsamples, dimension)`` with prescribed probability distribution.

        """

        x = np.zeros_like(z)

        if isinstance(dist_object, JointInd):
            if all(hasattr(m, 'icdf') for m in dist_object.marginals):
                for j in range(len(dist_object.marginals)):
                    f_i = dist_object.marginals[j].icdf
                    x[:, j] = np.atleast_2d(f_i(stats.norm.cdf(z[:, j]).T))
        elif isinstance(dist_object, DistributionContinuous1D):
            f_i = dist_object.icdf
            x = np.atleast_2d(f_i(stats.norm.cdf(z))).T
        elif isinstance(dist_object, list):
            _, n = np.shape(z)
            for j in range(n):
                f_i = dist_object[j].icdf
                x[:, j] = np.atleast_2d(f_i(stats.norm.cdf(z[:, j]).T))
        return x

    @staticmethod
    def jacobian_u_x(dist_object, z, x, cov):
        """
        This is a method to calculate the jacobian of the transformation :math:`\mathbf{J}_{\mathbf{ux}}`.

        This is a static method, part of the ``Nataf`` class.

        **Inputs:**

        * **u** (`ndarray`):
            Standard normal vector of shape ``(nsamples, dimension)``.

        * **x** (`ndarray`):
            Random vector of shape ``(nsamples, dimension)``.

        * **dist_object** ((list of) ``Distribution`` object(s)):
                    Probability distribution of each random variable. Must be an object of type
                    ``DistributionContinuous1D`` or ``JointInd``.

        * **cov** (`ndarray`):
        The covariance  matrix of shape ``(dimension, dimension)``.

        **Outputs:**

        * **jacobian_u_to_x** (`ndarray`):
            Matrix of shape ``(dimension, dimension)``.

        """
        from scipy.linalg import cholesky
        h = cholesky(cov, lower=True)
        m, n = np.shape(z)
        y = np.zeros(shape=(n, n))
        jacobian_u_to_x = [None] * m
        for i in range(m):
            for j in range(n):
                xi = np.array([x[i, j]])
                zi = np.array([z[i, j]])
                y[j, j] = dist_object[j].pdf(xi) / stats.norm.pdf(zi)
            jacobian_u_to_x[i] = np.linalg.solve(h, y)

        return jacobian_u_to_x


class Uncorrelate:
    """
    Remove correlation from correlated standard normal random variables.

    **Inputs:**

    * **samples** (`ndarray`):
        Correlated standard normal vector of shape ``(nsamples, dimension)``.

    * **cov** (`ndarray`):
        The correlation matrix. Must be an ``ndarray`` of shape ``(u.shape[1], u.shape[1])``.

    **Outputs:**

    * **u** (`ndarray`):
        Correlated standard normal vector of shape ``(nsamples, dimension)``.

    """

    def __init__(self, samples, corr):
        self.z = samples
        from scipy.linalg import cholesky
        h = cholesky(corr, lower=True)
        self.u = np.dot(np.linalg.inv(h), self.z.T).T


class Correlate:
    """
    Induce correlation to uncorrelated standard normal random variables.

    **Inputs:**

    * **samples** (`ndarray`):
        Uncorrelated standard normal vector of shape ``(nsamples, dimension)``.

    * **cov** (`ndarray`):
        The correlation matrix. Must be an ``ndarray`` of shape ``(u.shape[1], u.shape[1])``.

    **Outputs:**

    * **z** (`ndarray`):
        Correlated standard normal vector of shape ``(nsamples, dimension)``.

    """

    def __init__(self, samples, cov):
        self.u = samples
        self.corr = cov
        from scipy.linalg import cholesky
        h = cholesky(cov, lower=True)
        self.z = np.dot(h, self.u.T).T
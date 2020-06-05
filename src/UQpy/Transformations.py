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


class Isoprobabilistic:
    """
    Transform random variables  using the isoprobabilistic transformations

    This is the parent class to all isoprobabilistic transformation algorithms.

    **Inputs:**

    * **samples_x** (`ndarray`):
            Random vector of shape ``(nsamples, dimension)`` with prescribed probability distribution, with
            ``(nsamples, dimension) = samples_x.shape``.

    * **samples_y** (`ndarray`):
            Uncorrelated standard random vector of shape ``(nsamples, dimension)``
            ``(nsamples, dimension) = samples_x.shape``.

    * **dist_object** ((list of ) ``Distribution`` object(s)):
                    Probability distribution of each random variable. Must be an object of type
                    ``DistributionContinuous1D`` or ``JointInd``.

    * **corr_x** (`ndarray`):
        The correlation  matrix (:math:`\mathbf{C_X}`) of the random vector **X** .

    * **corr_z** (`ndarray`):
        The correlation  matrix (:math:`\mathbf{C_Z}`) of the standard normal random vector **Z** .

        Default: The ``identity`` matrix.

    """

    def __init__(self, dist_object, samples_x, samples_y, corr_x, corr_z):

        if isinstance(dist_object, list):
            for i in range(len(dist_object)):
                if not isinstance(dist_object[i], (DistributionContinuous1D, JointInd)):
                    raise TypeError('UQpy: A  ``DistributionContinuous1D`` or ``JointInd`` object must be provided.')
        else:
            if not isinstance(dist_object, (DistributionContinuous1D, JointInd)):
                raise TypeError('UQpy: A  ``DistributionContinuous1D``  or ``JointInd`` object must be provided.')

        self.dist_object = dist_object
        self.samples_x = samples_x
        self.samples_y = samples_y
        self.corr_x = corr_x
        self.corr_z = corr_z


class Nataf(Isoprobabilistic):
    """
    A class to perform the Nataf transformation.
    This is a an child class of the ``Isoprobabilistic`` class.

    **Inputs:**

    The ``Nataf`` class has the inputs: * **samples_x**, * **corr_x** and * **dist_object** as the ``Isoprob`` class
    plus:

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

    * **Y** (`ndarray`):
        Independent standard normal vector of shape ``(nsamples, dimension)``.

    * **corr_z** (`ndarray`):
        Distorted correlation matrix (:math:`\mathbf{C_Z}`) of the standard normal vector **Z**.

    * **Jxy** (`ndarray`):
        The jacobian of the transformation of shape ``(dimension, dimension)``.

    **Methods:**
    """

    def __init__(self, samples_x, dist_object, corr_x=None, beta=1.0, itam_error1=0.001,
                 itam_error2=0.01, corr_z=None, samples_y=None):

        super().__init__(dist_object, samples_x, samples_y, corr_x, corr_z)

        self.beta = beta
        self.itam_error1 = itam_error1
        self.itam_error2 = itam_error2
        self.corr_x = corr_x
        self.dist_object = dist_object
        self.samples_x = samples_x

        if corr_x is None:
            self.corr_x = np.eye(self.samples_x.shape[1])

        if np.all(np.equal(self.corr_x, np.eye(self.samples_x.shape[1]))):
            self.corr_z = self.corr_x
            self.samples_y = self.transform_x_to_y(self.samples_x, self.dist_object, self.corr_z)
            self.Jxy = np.eye(self.samples_x.shape[1])
        else:
            self.corr_z = self.distortion_x_to_z(self.dist_object, self.corr_x, self.beta, self.itam_error1,
                                                 self.itam_error2)
            self.samples_y = self.transform_x_to_y(self.samples_x, self.dist_object, self.corr_z)
            self.Jxy = self.jacobian_x_y(self.dist_object, self.samples_x, self.samples_y, self.corr_z)

    @staticmethod
    def distortion_x_to_z(dist_object, corr_x,  beta=1.0, itam_error1=0.001, itam_error2=0.01):
        """
        This is a method to calculate the correlation matrix :math:`\mathbf{C_Z}` of the standard normal random vector
        :math:`\mathbf{z}`  given the correlation matrix :math:`\mathbf{C_x}` of the random vector :math:`\mathbf{x}`
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
        from UQpy.Utilities import itam
        cov_distorted = itam(dist_object, corr_x, beta, itam_error1, itam_error2)
        return cov_distorted

    @staticmethod
    def transform_x_to_y(samples_x, dist_object, corr_z):
        """
        This is a method to transform a vector :math:`\mathbf{x}` of  samples with marginal distributions
        :math:`f_i(x_i)` and cumulative distributions :math:`F_i(x_i)` to a vector :math:`\mathbf{z}` of standard normal
        samples  according to: :math:`Z_{i}=\Phi^{-1}(F_i(X_{i}))`, where :math:`\Phi` is the cumulative
        distribution function of a standard  normal variable.

        This is a static method, part of the ``Nataf`` class.

        **Inputs:**

        * **samples_x** (`ndarray`):
            Random vector of shape ``(nsamples, dimension)`` with prescribed probability distributions.

        * **dist_object** ((list of) ``Distribution`` object(s)):
                    Probability distribution of each random variable. Must be an object of type
                    ``DistributionContinuous1D`` or ``JointInd``.
        **Outputs:**

        * **samples_z** (`ndarray`):
            Standard normal random vector of shape ``(nsamples, dimension)``.

        """
        samples_z = np.zeros_like(samples_x)

        if isinstance(dist_object, JointInd):
            if all(hasattr(m, 'cdf') for m in dist_object.marginals):
                for j in range(len(dist_object.marginals)):
                    samples_z[:, j] = stats.norm.ppf(dist_object.marginals[j].cdf(np.atleast_2d(samples_x[:, j]).T))
        elif isinstance(dist_object, DistributionContinuous1D):
            f_i = dist_object.cdf
            samples_z = np.atleast_2d(stats.norm.ppf(f_i(samples_x))).T
        else:
            m, n = np.shape(samples_x)
            for j in range(n):
                f_i = dist_object[j].cdf
                samples_z[:, j] = stats.norm.ppf(f_i(np.atleast_2d(samples_x[:, j]).T))
        print(corr_z)
        samples_y = Nataf.decorrelate(samples_z, corr_z)

        return samples_y

    @staticmethod
    def jacobian_x_y(dist_object, samples_x, samples_y, corr_z):
        """
        This is a method to calculate the jacobian of the transformation :math:`\mathbf{J}_{\mathbf{xy}}`.

        This is a static method, part of the ``Nataf`` class.

        **Inputs:**

        * **samples_z** (`ndarray`):
            Standard normal vector of shape ``(nsamples, dimension)``.

        * **samples_x** (`ndarray`):
            Random vector of shape ``(nsamples, dimension)``.

        * **dist_object** ((list of) ``Distribution`` object(s)):
                    Probability distribution of each random variable. Must be an object of type
                    ``DistributionContinuous1D`` or ``JointInd``.

        * **corr_z** (`ndarray`):
        The covariance  matrix of shape ``(dimension, dimension)``.

        **Outputs:**

        * **jacobian_x_to_u** (`ndarray`):
            Matrix of shape ``(dimension, dimension)``.

        """
        samples_z = InvNataf.correlate(samples_y, corr_z)
        from scipy.linalg import cholesky
        h = cholesky(corr_z, lower=True)
        m, n = np.shape(samples_z)
        y = np.zeros(shape=(n, n))
        jacobian_x_to_y = [None] * m
        for i in range(m):
            for j in range(n):
                xi = np.array([samples_x[i, j]])
                zi = np.array([samples_z[i, j]])
                y[j, j] = stats.norm.pdf(zi) / dist_object[j].pdf(xi)
            jacobian_x_to_y[i] = np.linalg.solve(y, h)

        return jacobian_x_to_y

    @staticmethod
    def decorrelate(samples_z, corr_z):

        """
        Remove correlation from standard normal random variables.

        **Inputs:**

        * **samples_z** (`ndarray`):
            Correlated standard normal vector of shape ``(nsamples, dimension)``.

        * **corr_z** (`ndarray`):
            The correlation matrix. Must be an ``ndarray`` of shape ``(u.shape[1], u.shape[1])``.

        **Outputs/Returns:**

        * **samples_y** (`ndarray`):
            Uncorrelated standard normal vector of shape ``(nsamples, dimension)``.

        """

        from scipy.linalg import cholesky
        h = cholesky(corr_z, lower=True)
        samples_y = np.linalg.solve(h, samples_z.T).T

        return samples_y


class InvNataf(Isoprobabilistic):
    """
    A class perform the inverse Nataf transformation, i.e. transform independent standard normal variables to
    arbitrarily distributed random variables.

    This is a an child class of the ``Isoprobabilistic`` class.

    **Inputs:**

    The ``InvNataf`` class has the inputs: * **samples_y**, * **corr_z** and * **dist_object** as the ``Isoprob`` class.

    **Attributes:**

    * **samples_x** (`ndarray`):
        Independent standard normal vector of shape ``(nsamples, dimension)``.

    * **corr_x** (`ndarray`):
        Distorted correlation matrix of the random vector **X** of shape ``(dimension, dimension)``.

    * **Jyx** (`ndarray`):
        The jacobian of the transformation of shape ``(dimension, dimension)``.

    **Methods:**

    """

    def __init__(self, samples_y, dist_object, corr_z=None, corr_x=None, samples_x=None):

        super().__init__(dist_object, samples_x, samples_y,  corr_x, corr_z)

        self.corr_z = corr_z
        self.dist_object = dist_object
        self.samples_y = samples_y

        if corr_z is None:
            self.corr_z = np.eye(self.samples_y.shape[1])

        if np.all(np.equal(self.corr_z, np.eye(self.samples_y.shape[1]))):
            self.corr_x = self.corr_z
            self.samples_x = self.transform_y_to_x(self.samples_y, self.dist_object, self.corr_z)
            self.Jyx = np.eye(self.samples_x.shape[1])

        else:
            self.corr_x = self.distortion_z_to_x(self.dist_object, self.corr_z)
            self.samples_x = self.transform_y_to_x(self.samples_y, self.dist_object, self.corr_z)
            self.Jyx = self.jacobian_y_x(self.dist_object, self.samples_y, self.samples_x, corr_z)

    @staticmethod
    def distortion_z_to_x(dist_object, corr_z):
        """
        This is a method to calculate the correlation matrix :math:`\mathbf{C_x}` of the random vector
        :math:`\mathbf{x}`  given the correlation matrix :math:`\mathbf{C_z}` of the standard normal random vector
        :math:`\mathbf{z}` using the `correlation_distortion` method (see ``Utilities`` class).

        This is a static method, part of the ``InvNataf`` class.

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

    @staticmethod
    def transform_y_to_x(samples_y, dist_object, corr_z):
        """
        This is a method to transform a vector :math:`\mathbf{y}` of  standard normal samples to a vector
        :math:`\mathbf{x}` of  samples with marginal distributions :math:`f_i(x_i)` and cumulative distributions
        :math:`F_i(x_i)` according to: :math:`Z_{i}=F_i^{-1}(\Phi(Z_{i}))`, where :math:`\Phi` is the cumulative
        distribution function of a standard  normal variable.

        This is a static method, part of the ``InvNataf`` class.

        **Inputs:**

        * **samples_y** (`ndarray`):
            Standard normal vector of shape ``(nsamples, dimension)``.

        * **dist_object** ((list of) ``Distribution`` object(s)):
                    Probability distribution of each random variable. Must be an object of type
                    ``DistributionContinuous1D`` or ``JointInd``.

        **Outputs:**

        * **samples_x** (`ndarray`):
            Random vector of shape ``(nsamples, dimension)`` with prescribed probability distribution.

        """
        samples_z = InvNataf.correlate(samples_y, corr_z)
        samples_x = np.zeros_like(samples_y)

        if isinstance(dist_object, JointInd):
            if all(hasattr(m, 'icdf') for m in dist_object.marginals):
                for j in range(len(dist_object.marginals)):
                    f_i = dist_object.marginals[j].icdf
                    samples_x[:, j] = np.atleast_2d(f_i(stats.norm.cdf(samples_z[:, j]).T))
        elif isinstance(dist_object, DistributionContinuous1D):
            f_i = dist_object.icdf
            samples_x = np.atleast_2d(f_i(stats.norm.cdf(samples_z))).T
        elif isinstance(dist_object, list):
            for j in range(samples_y.shape[1]):
                f_i = dist_object[j].icdf
                samples_x[:, j] = np.atleast_2d(f_i(stats.norm.cdf(samples_z[:, j]).T))
        return samples_x

    @staticmethod
    def jacobian_y_x(dist_object, samples_y, samples_x, corr_z):
        """
        This is a method to calculate the jacobian of the transformation :math:`\mathbf{J}_{\mathbf{yx}}`.

        This is a static method, part of the ``InvNataf`` class.

        **Inputs:**

        * **samples_z** (`ndarray`):
            Standard normal vector of shape ``(nsamples, dimension)``.

        * **samples_x** (`ndarray`):
            Random vector of shape ``(nsamples, dimension)``.

        * **dist_object** ((list of) ``Distribution`` object(s)):
                    Probability distribution of each random variable. Must be an object of type
                    ``DistributionContinuous1D`` or ``JointInd``.

        * **corr_z** (`ndarray`):
        The covariance  matrix of shape ``(dimension, dimension)``.

        **Outputs:**

        * **jacobian_y_to_x** (`ndarray`):
            Matrix of shape ``(dimension, dimension)``.

        """
        samples_z = InvNataf.correlate(samples_y, corr_z)
        from scipy.linalg import cholesky
        h = cholesky(corr_z, lower=True)
        m, n = np.shape(samples_z)
        y = np.zeros(shape=(n, n))
        jacobian_y_to_x = [None] * m
        for i in range(m):
            for j in range(n):
                xi = np.array([samples_x[i, j]])
                zi = np.array([samples_z[i, j]])
                y[j, j] = dist_object[j].pdf(xi) / stats.norm.pdf(zi)
            jacobian_y_to_x[i] = np.linalg.solve(h, y)

        return jacobian_y_to_x

    @staticmethod
    def correlate(samples_y, corr_z):
        """
        Induce correlation to uncorrelated standard normal random variables.

        This is a static method, part of the ``InvNataf`` class.

        **Inputs:**

        * **samples_y** (`ndarray`):
            Uncorrelated standard normal vector of shape ``(nsamples, dimension)``.

        * **corr_z** (`ndarray`):
            The correlation matrix. Must be an ``ndarray`` of shape ``(u.shape[1], u.shape[1])``.

        **Outputs/Returns:**

        * **samples_z** (`ndarray`):
            Correlated standard normal vector of shape ``(nsamples, dimension)``.

        """

        from scipy.linalg import cholesky
        h = cholesky(corr_z, lower=True)
        samples_z = np.dot(h, samples_y.T).T

        return samples_z




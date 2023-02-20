import logging
from typing import Union, Annotated
from beartype import beartype
from beartype.vale import Is

from UQpy.distributions import *
import numpy as np
import scipy.stats as stats
from UQpy.utilities.Utilities import nearest_psd, calculate_gauss_quadrature_2d
from UQpy.utilities.Utilities import bi_variate_normal_pdf
from scipy.linalg import cholesky
from UQpy.utilities.ValidationTypes import PositiveInteger, NumpyFloatArray

DistributionList = Annotated[
    list[Distribution],
    Is[lambda lst: all(isinstance(item, Distribution) for item in lst)],
]


class Nataf:

    @beartype
    def __init__(
            self,
            distributions: Union[Distribution, DistributionList],
            samples_x: Union[None, np.ndarray] = None,
            samples_z: Union[None, np.ndarray] = None,
            jacobian: bool = False,
            corr_z: Union[None, np.ndarray] = None,
            corr_x: Union[None, np.ndarray] = None,
            itam_beta: Union[float, int] = 1.0,
            itam_threshold1: Union[float, int] = 0.001,
            itam_threshold2: Union[float, int] = 0.1,
            itam_max_iter: int = 100,
            n_gauss_points: int = 128
    ):
        """
        Transform random variables using the Nataf or Inverse Nataf transformation

        :param distributions: Probability distribution of each random variable. Must be an object of type
         :class:`.DistributionContinuous1D` or :class:`.JointIndependent`.
        :param samples_x: Random vector of shape ``(nsamples, n_dimensions)`` with prescribed probability
         distributions.
         If `samples_x` is provided, the :class:`.Nataf` class transforms them to `samples_z`.
        :param samples_z: Standard normal random vector of shape ``(nsamples_, n_dimensions)``
         If `samples_z` is provided, the :class:`.Nataf` class transforms them to `samples_x`.
        :param jacobian: A boolean whether to return the jacobian of the transformation. Default: :any:`False`
        :param corr_z: The correlation  matrix (:math:`\mathbf{C_Z}`) of the standard normal random vector **Z** .
         Default: The identity matrix.
         If ``corr_z`` is specified, the :class:`.Nataf` class computes the correlation distortion integral above to
         solve for ``corr_x``.
        :param corr_x: The correlation  matrix (:math:`\mathbf{C_X}`) of the random vector **X** .
         Default: The identity matrix.
         If ``corr_x`` is specified, the :class:`.Nataf` class invokes the ITAM to compute ``corr_z``.
        :param itam_beta: A parameter selected to optimize convergence speed and desired accuracy of the ITAM method.
         Default: :math:`1.0`
        :param itam_threshold1: If ``corr_x`` is specified, this specifies the threshold value for the error in the
         `ITAM` method (see :py:mod:`.utilities` module) given by
         :math:`\epsilon_1 = ||\mathbf{C_X}^{target} - \mathbf{C_X}^{computed}||`
         Default: :math:`0.001`
        :param itam_threshold2: If ``corr_x`` is specified, this specifies the threshold value for the error difference
         between iterations in the `ITAM` method (see :py:mod:`.utilities` module) given by
         :math:`\epsilon_1^{i} - \epsilon_1^{i-1}`
         for iteration :math:`i`.
         Default: :math:`0.01`
        :param itam_max_iter: Maximum number of iterations for the ITAM method. Default: :math:`100`
        :param n_gauss_points: The number of integration points used for the numerical integration of the
         correlation  matrix (:math:`\mathbf{C_Z}`) of the standard normal random vector **Z**
        """
        self.n_dimensions = 0
        if isinstance(distributions, list):
            for i in range(len(distributions)):
                self.update_dimensions(distributions[i])
        else:
            self.update_dimensions(distributions)
        self.dist_object = distributions
        self.samples_x: NumpyFloatArray = samples_x
        """Random vector of shape ``(nsamples, n_dimensions)`` with prescribed probability distributions."""
        self.samples_z: NumpyFloatArray = samples_z
        """Standard normal random vector of shape ``(nsamples, n_dimensions)``"""
        self.jacobian = jacobian
        self.jzx: NumpyFloatArray = None
        """The Jacobian of the transformation of shape ``(n_dimensions, n_dimensions)``."""
        self.jxz: NumpyFloatArray = None
        """The Jacobian of the transformation of shape ``(n_dimensions, n_dimensions)``."""
        self.itam_max_iter = itam_max_iter
        self.itam_beta = float(itam_beta)
        self.itam_threshold1 = float(itam_threshold1)
        self.itam_threshold2 = float(itam_threshold2)
        self.logger = logging.getLogger(__name__)

        if corr_x is None and corr_z is None:
            self.corr_x: NumpyFloatArray = np.eye(self.n_dimensions)
            """Distorted correlation matrix (:math:`\mathbf{C_X}`) of the random vector **X**."""
            self.corr_z: NumpyFloatArray = np.eye(self.n_dimensions)
            """Distorted correlation matrix (:math:`\mathbf{C_Z}`) of the standard normal vector **Z**."""
        elif corr_x is not None:
            self.corr_x = corr_x
            if np.all(np.equal(self.corr_x, np.eye(self.n_dimensions))):
                self.corr_z = self.corr_x
            elif all(isinstance(x, Normal) for x in distributions):
                self.corr_z = self.corr_x
            else:
                self.corr_z, self.itam_error1, self.itam_error2 = \
                    self.itam(self.dist_object, self.corr_x, self.itam_max_iter, self.itam_beta,
                              self.itam_threshold1, self.itam_threshold2, )
        elif corr_z is not None:
            self.corr_z = corr_z
            if np.all(np.equal(self.corr_z, np.eye(self.n_dimensions))):
                self.corr_x = self.corr_z
            elif all(isinstance(x, Normal) for x in distributions):
                self.corr_x = self.corr_z
            else:
                self.corr_x = self.distortion_z2x(self.dist_object, self.corr_z, n_gauss_points=n_gauss_points)

        self.H: NumpyFloatArray = cholesky(self.corr_z, lower=True)
        """The lower triangular matrix resulting from the Cholesky decomposition of the correlation matrix
        :math:`\mathbf{C_Z}`."""

        if self.samples_x is not None or self.samples_z is not None:
            self.run(self.samples_x, self.samples_z, self.jacobian)

    @beartype
    def run(
            self,
            samples_x: Union[None, np.ndarray] = None,
            samples_z: Union[None, np.ndarray] = None,
            jacobian: bool = False,
    ):
        """
        Execute the Nataf transformation or its inverse.
        If `samples_x` is provided, the :meth:`run` method performs the Nataf transformation. If `samples_z` is
        provided, the :meth:`run` method performs the inverse Nataf transformation.

        :param samples_x: Random vector **X**  with prescribed probability distributions or standard normal random
         vector **Z** of shape :code:`(nsamples, n_dimensions)`.
        :param samples_z: Standard normal random vector of shape ``(nsamples, n_dimensions)``
         If `samples_z` is provided, the :class:`.Nataf` class transforms them to `samples_x`.
        :param jacobian: The jacobian of the transformation of shape ``(n_dimensions, n_dimensions)``. Default:
         :any:`False`
        """
        self.jacobian = jacobian

        if samples_x is not None:
            if len(samples_x.shape) != 2:
                self.samples_x = np.atleast_2d(samples_x).T
            else:
                self.samples_x = samples_x
            if not self.jacobian:
                self.samples_z = self._transform_x2z(self.samples_x)
            elif self.jacobian:
                self.samples_z, self.jxz = self._transform_x2z(self.samples_x, jacobian=self.jacobian)

        if samples_z is not None:
            if len(samples_z.shape) != 2:
                self.samples_z = np.atleast_2d(samples_z).T
            else:
                self.samples_z = samples_z
            if not self.jacobian:
                self.samples_x = self._transform_z2x(self.samples_z)
            elif self.jacobian:
                self.samples_x, self.jzx = self._transform_z2x(self.samples_z, jacobian=self.jacobian)

    @staticmethod
    def itam(
            distributions: Union[
                DistributionContinuous1D,
                JointIndependent,
                list[Union[DistributionContinuous1D, JointIndependent]]],
            corr_x,
            itam_max_iter: int = 100,
            itam_beta: Union[float, int] = 1.0,
            itam_threshold1: Union[float, int] = 0.001,
            itam_threshold2: Union[float, int] = 0.01,
    ):
        """
        Calculate the correlation matrix :math:`\mathbf{C_Z}` of the standard normal random vector
        :math:`\mathbf{Z}` given the correlation matrix :math:`\mathbf{C_X}` of the random vector :math:`\mathbf{X}`
        using the `ITAM` method :cite:`StochasticProcess13`.

        :param distributions:  Probability distribution of each random variable. Must be an object of type
         :class:`.DistributionContinuous1D` or :class:`.JointIndependent`.
         `distributions` must have a ``cdf`` method.
        :param corr_x: The correlation  matrix (:math:`\mathbf{C_X}`) of the random vector **X** .
         Default: The ``identity`` matrix.
        :param itam_max_iter: Maximum number of iterations for the ITAM method. Default: :math:`100`
        :param itam_beta: A parameters selected to optimize convergence speed and desired accuracy of the ITAM method
         (see :cite:`Nataf2`). Default: :math:`1.0`
        :param itam_threshold1: A threshold value for the relative difference between the non-Gaussian correlation
         function and the underlying Gaussian. Default: :math:`0.001`
        :param itam_threshold2: If ``corr_x`` is specified, this specifies the threshold value for the error difference
         between iterations in the `ITAM` method (see :py:mod:`.utilities` module) given by:
         :math:`\epsilon_1^{i} - \epsilon_1^{i-1}`
         for iteration :math:`i`. Default: :math:`0.01`
        :return: Distorted correlation matrix (:math:`\mathbf{C_Z}`) of the standard normal vector **Z**,
         List of ITAM errors for each iteration, List of ITAM difference errors for each iteration
        """
        # Initial Guess
        corr_z0 = corr_x
        corr_z = np.zeros_like(corr_z0)
        # Iteration Condition
        itam_error1 = []
        itam_error2 = []
        itam_error1.append(100.0)
        itam_error2.append(abs(itam_error1[0] - 0.1) / 0.1)

        logger = logging.getLogger(__name__)

        logger.info("UQpy: Initializing Iterative Translation Approximation Method (ITAM)")

        for k in range(itam_max_iter):
            error0 = itam_error1[k]

            corr0 = Nataf.distortion_z2x(distributions, corr_z0)

            max_ratio = np.amax(np.ones((len(corr_x), len(corr_x))) / abs(corr_z0))

            corr_z = np.nan_to_num((corr_x / corr0) ** itam_beta * corr_z0)

            # Do not allow off-diagonal correlations to equal or exceed one
            corr_z[corr_z < -1.0] = (max_ratio + 1) / 2 * corr_z0[corr_z < -1.0]
            corr_z[corr_z > 1.0] = (max_ratio + 1) / 2 * corr_z0[corr_z > 1.0]

            corr_z = np.array(nearest_psd(corr_z))

            corr_z0 = corr_z.copy()

            itam_error1.append(np.linalg.norm(corr_x - corr0))
            itam_error2.append(abs(itam_error1[-1] - error0) / error0)

            logger.info("UQpy: ITAM iteration number %i", k)
            logger.info("UQpy: Current error, %e", (itam_error1[-1], itam_error2[-1]))

            if itam_error1[k] <= itam_threshold1 and itam_error2[k] <= itam_threshold2:
                break

        logger.info("UQpy: ITAM Done.")

        return corr_z, itam_error1, itam_error2

    @staticmethod
    def distortion_z2x(distributions: Union[Distribution, list[Distribution]], corr_z: np.ndarray,
                       n_gauss_points: int = 1024):
        """
        This is a method to calculate the correlation matrix :math:`\mathbf{C_x}` of the random vector
        :math:`\mathbf{x}`  given the correlation matrix :math:`\mathbf{C_z}` of the standard normal random vector
        :math:`\mathbf{z}`.

        :param distributions: This is a method to calculate the correlation matrix :math:`\mathbf{C_x}` of the random vector
         :math:`\mathbf{x}`  given the correlation matrix :math:`\mathbf{C_z}` of the standard normal random vector
         :math:`\mathbf{z}`.
         This method is part of the :class:`.Nataf` class.
        :param corr_z: The correlation  matrix (:math:`\mathbf{C_z}`) of the standard normal vector **Z** .
         Default: The ``identity`` matrix.
        :param n_gauss_points: The number of integration points used for the numerical integration of the
         correlation  matrix (:math:`\mathbf{C_Z}`) of the standard normal random vector **Z**
        :return: Distorted correlation matrix (:math:`\mathbf{C_X}`) of the random vector **X**.
        """
        logger = logging.getLogger(__name__)
        z_max = 8
        z_min = -z_max
        ng = n_gauss_points

        eta, w2d, xi = calculate_gauss_quadrature_2d(ng, z_max, z_min)

        corr_x = np.ones_like(corr_z)
        logger.info("UQpy: Computing Nataf correlation distortion...")

        is_joint = isinstance(distributions, JointIndependent)
        marginals = distributions.marginals if is_joint else distributions
        corr_x = Nataf.calculate_corr_x(corr_x, corr_z, marginals, eta, w2d, xi, is_joint)
        return corr_x

    @staticmethod
    def calculate_corr_x(corr_x, corr_z, marginals, eta, w2d, xi, is_joint):
        if all(hasattr(m, "moments") for m in marginals) and all(
                hasattr(m, "icdf") for m in marginals
        ):
            for i in range(len(marginals)):
                i_cdf_i = marginals[i].icdf
                mi = marginals[i].moments()
                if not (np.isfinite(mi[0]) and np.isfinite(mi[1])):
                    raise RuntimeError("UQpy: The marginal distributions need to have finite mean and variance.")
                for j in range(i + 1, len(marginals)):
                    i_cdf_j = marginals[j].icdf
                    mj = marginals[j].moments()
                    if not (np.isfinite(mj[0]) and np.isfinite(mj[1])):
                        raise RuntimeError("UQpy: The marginal distributions need to have finite mean and variance.")
                    term1 = mj[0] ** 2 if is_joint else mj[0]
                    term2 = mi[0] ** 2 if is_joint else mi[0]
                    tmp_f_xi = i_cdf_j(np.atleast_2d(stats.norm.cdf(xi)).T) - term1
                    tmp_f_eta = i_cdf_i(np.atleast_2d(stats.norm.cdf(eta)).T) - term2

                    phi2 = bi_variate_normal_pdf(xi, eta, corr_z[i, j])

                    corr_x[i, j] = (1 / (np.sqrt(mj[1]) * np.sqrt(mi[1])) * np.sum(tmp_f_xi * tmp_f_eta * w2d * phi2))
                    corr_x[j, i] = corr_x[i, j]
        return corr_x

    def _transform_x2z(self, samples_x, jacobian=False):
        """
        This is a method to transform a vector :math:`\mathbf{x}` of  samples with marginal distributions
        :math:`f_i(x_i)` and cumulative distributions :math:`F_i(x_i)` to a vector :math:`\mathbf{z}` of standard normal
        samples  according to: :math:`Z_{i}=\Phi^{-1}(F_i(X_{i}))`, where :math:`\Phi` is the cumulative
        distribution function of a standard  normal variable.

        :param samples_x: Random vector of shape ``(nsamples, n_dimensions)`` with prescribed probability distributions.
        :param jacobian: A boolean whether to return the jacobian of the transformation.
         Default: False
        :return: Standard normal random vector of shape ``(nsamples, n_dimensions)``, the jacobian of the transformation
         of shape ``(n_dimensions, n_dimensions)``.
        """

        m, n = np.shape(samples_x)
        samples_z = None

        if isinstance(self.dist_object, JointIndependent):
            if all(hasattr(m, "cdf") for m in self.dist_object.marginals):
                samples_z = np.zeros_like(samples_x)
                for j in range(len(self.dist_object.marginals)):
                    samples_z[:, j] = stats.norm.ppf(self.dist_object.marginals[j].cdf(samples_x[:, j]))
        elif isinstance(self.dist_object, DistributionContinuous1D):
            samples_z = stats.norm.ppf(self.dist_object.cdf(samples_x))
        else:
            samples_z = np.zeros_like(samples_x)
            for j in range(n):
                samples_z[:, j] = stats.norm.ppf(
                    self.dist_object[j].cdf(samples_x[:, j])
                )

        if not jacobian:
            return samples_z
        jac = np.zeros(shape=(n, n))
        jxz = [None] * m
        for i in range(m):
            for j in range(n):
                xi = np.array([samples_x[i, j]])
                zi = np.array([samples_z[i, j]])
                jac[j, j] = stats.norm.pdf(zi) / self.dist_object[j].pdf(xi)
            jxz[i] = np.linalg.solve(jac, self.H)

        return samples_z, jxz

    def _transform_z2x(self, samples_z, jacobian=False):
        """
        This is a method to transform a standard normal vector :math:`\mathbf{z}` to a vector
        :math:`\mathbf{x}` of samples with marginal distributions :math:`f_i(x_i)` and cumulative distributions
        :math:`F_i(x_i)` to samples  according to: :math:`Z_{i}=\Phi^{-1}(F_i(X_{i}))`, where :math:`\Phi` is the
        cumulative distribution function of a standard  normal variable.

        :param samples_z: Standard normal random vector of shape ``(nsamples, n_dimensions)``
        :param jacobian: A boolean whether to return the jacobian of the transformation. Default: False
        :return: Random vector of shape ``(nsamples, n_dimensions)`` with prescribed probability distributions,
         The jacobian of the transformation of shape ``(n_dimensions, n_dimensions)``.
        """
        m, n = np.shape(samples_z)
        h = cholesky(self.corr_z, lower=True)
        # samples_z = (h @ samples_y.T).T
        samples_x = np.zeros_like(samples_z)
        if isinstance(self.dist_object, JointIndependent):
            if all(hasattr(m, "icdf") for m in self.dist_object.marginals):
                for j in range(len(self.dist_object.marginals)):
                    samples_x[:, j] = self.dist_object.marginals[j].icdf(stats.norm.cdf(samples_z[:, j]))

        elif isinstance(self.dist_object, DistributionContinuous1D):
            samples_x = self.dist_object.icdf(stats.norm.cdf(samples_z))
        elif isinstance(self.dist_object, list):
            for j in range(samples_x.shape[1]):
                samples_x[:, j] = self.dist_object[j].icdf(stats.norm.cdf(samples_z[:, j]))

        if not jacobian:
            return samples_x
        jac = np.zeros(shape=(n, n))
        jzx = [None] * m
        for i in range(m):
            for j in range(n):
                xi = np.array([samples_x[i, j]])
                zi = np.array([samples_z[i, j]])
                jac[j, j] = self.dist_object[j].pdf(xi) / stats.norm.pdf(zi)
            jzx[i] = np.linalg.solve(h, jac)

        return samples_x, jzx

    @beartype
    def rvs(self, nsamples: PositiveInteger):
        """
        Generate realizations from the joint pdf of the random vector **X**.

        :param nsamples: Number of samples to generate.
        :return: Random vector in the parameter space of shape ``(nsamples, n_dimensions)``.
        """
        h = cholesky(self.corr_z, lower=True)
        n = int(nsamples)
        m = np.size(self.dist_object)
        y = np.random.randn(nsamples, m)
        z = np.dot(h, y.T).T
        samples_x = np.zeros([n, m])
        for i in range(m):
            samples_x[:, i] = self.dist_object[i].icdf(stats.norm.cdf(z[:, i]))
        return samples_x

    def update_dimensions(self, dist_object):
        if isinstance(dist_object, DistributionContinuous1D):
            self.n_dimensions += 1
        elif isinstance(dist_object, JointIndependent):
            self.n_dimensions += len(dist_object.marginals)
        else:
            raise TypeError("UQpy: A  ``DistributionContinuous1D``  or ``JointIndependent`` object must be provided.")

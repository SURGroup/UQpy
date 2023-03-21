import logging
from typing import Union
import numpy as np
import scipy.stats as stats
from beartype import beartype

from UQpy.run_model.RunModel import RunModel
from UQpy.transformations import *
from UQpy.distributions import *
from UQpy.reliability.taylor_series.baseclass.TaylorSeries import TaylorSeries
from UQpy.utilities.ValidationTypes import PositiveInteger
from UQpy.transformations import Decorrelate
import warnings

warnings.filterwarnings('ignore')


class FORM(TaylorSeries):

    @beartype
    def __init__(
        self,
        distributions: Union[None, Distribution, list[Distribution]],
        runmodel_object: RunModel,
        seed_x: Union[list, np.ndarray] = None,
        seed_u: Union[list, np.ndarray] = None,
        df_step: Union[int, float] = 0.01,
        corr_x: Union[list, np.ndarray] = None,
        corr_z: Union[list, np.ndarray] = None,
        n_iterations: PositiveInteger = 100,
        tol1: Union[float, int] = None,
        tol2: Union[float, int] = None,
        tol3: Union[float, int] = None,
    ):
        """
        A class perform the First Order reliability Method. The :meth:`run` method of the :class:`.FORM` class can be invoked many
        times and each time the results are appended to the existing ones.
        This is a child class of the :class:`.TaylorSeries` class.

        :param distributions: Marginal probability distributions of each random variable. Must be an object of
         type :class:`.DistributionContinuous1D` or :class:`.JointIndependent`.
        :param runmodel_object: The computational model. It should be of type :class:`RunModel`.
        :param seed_u: The initial starting point in the uncorrelated standard normal space **U** for the `Hasofer-Lind`
         algorithm.
         Either `seed_u` or `seed_x` must be provided.
         Default: :code:`seed_u = (0, 0, ..., 0)`
         If either `seed_u` or `seed_x` is provided, then the :py:meth:`run` method will be executed automatically.
         Otherwise, the the :py:meth:`run` method must be executed by the user.
        :param seed_x: The initial starting point in the parameter space **X** for the `Hasofer-Lind` algorithm.
         Either `seed_u` or `seed_x` must be provided.
         If either `seed_u` or `seed_x` is provided, then the :py:meth:`run` method will be executed automatically.
         Otherwise, the the :py:meth:`run` method must be executed by the user.
        :param df_step: Finite difference step in standard normal space. Default: :math:`0.01`
        :param corr_z: Covariance matrix
         If `corr_z` is provided, it is the correlation matrix (:math:`\mathbf{C_Z}`) of the standard normal random
         vector **Z** .
         If `corr_x` is provided, it is the correlation matrix (:math:`\mathbf{C_X}`) of the random vector **X** .
        :param corr_z: Covariance matrix
         If `corr_x` is provided, it is the correlation matrix (:math:`\mathbf{C_X}`) of the random vector **X** .
         If `corr_z` is provided, it is the correlation matrix (:math:`\mathbf{C_Z}`) of the standard normal random
         vector **Z** .
         Default: `corr_z` is specified as the identity matrix.
        :param n_iterations: Maximum number of iterations for the `HLRF` algorithm. Default: :math:`100`
        :param tol1: Convergence threshold for criterion `e1` of the `HLRF` algorithm. Default: :math:`1.0e-3`
        :param tol2: Convergence threshold for criterion `e2` of the `HLRF` algorithm. Default: :math:`1.0e-3`
        :param tol3: Convergence threshold for criterion `e3` of the  `HLRF` algorithm. Default: :math:`1.0e-3`

        Any number of tolerances can be provided. Only the provided tolerances will be considered for the convergence
        of the algorithm. In case none of the tolerances is provided then they are considered equal to :math:`1e-3` and
        all are checked for the convergence.
        """
        if isinstance(distributions, list):
            self.dimension = len(distributions)
            self.dimension = 0
            for i in range(len(distributions)):
                if isinstance(distributions[i], DistributionContinuous1D):
                    self.dimension += 1
                elif isinstance(distributions[i], JointIndependent):
                    self.dimension += len(distributions[i].marginals)
                else:
                    raise TypeError(
                        "UQpy: A  ``DistributionContinuous1D`` or ``JointIndependent`` object must be "
                        "provided.")
        elif isinstance(distributions, DistributionContinuous1D):
            self.dimension = 1
        elif isinstance(distributions, JointIndependent):
            self.dimension = len(distributions.marginals)
        else:
            raise TypeError("UQpy: A  ``DistributionContinuous1D``  or ``JointIndependent`` object must be provided.")

        self.nataf_object = Nataf(distributions=distributions, corr_z=corr_z, corr_x=corr_x)

        self.corr_x = corr_x
        self.corr_z = corr_z
        self.dist_object = distributions
        self.n_iterations = n_iterations
        self.runmodel_object = runmodel_object
        self.tol1 = tol1
        self.tol2 = tol2
        self.tol3 = tol3
        self.seed_u = seed_u
        self.seed_x = seed_x

        self.logger = logging.getLogger(__name__)

        # Initialize output
        self.beta: float = None
        """Hasofer-Lind reliability index."""
        self.DesignPoint_U: list = None
        """Design point in the uncorrelated standard normal space U."""
        self.DesignPoint_X: list = None
        """Design point in the parameter space X."""
        self.alpha: float = None
        """Direction cosine."""
        self.failure_probability = None
        self.x = None
        self.g0 = None
        self.iterations: int = None
        """Number of model evaluations."""
        self.df_step = df_step
        self.error_record: list = None
        """Record of the error defined by criteria `e1, e2, e3`."""

        self.tol1 = tol1
        self.tol2 = tol2
        self.tol3 = tol3

        self.u_record: list = None
        """Record of all iteration points in the standard normal space **U**."""
        self.x_record: list = None
        """Record of all iteration points in the parameter space **X**."""
        self.g_record: list = None
        """Record of the performance function."""
        self.dg_u_record: list = None
        """Record of the model’s gradient in the standard normal space."""
        self.alpha_record: list = None
        """Record of the alpha (directional cosine)."""
        self.beta_record: list = None
        """Record of all Hasofer-Lind reliability index values."""
        self.jacobian_zx = None
        """Jacobian of the transformation from correlated standard normal space to the parameter space."""
        self.call = None

        if self.seed_u is not None:
            self.run(seed_u=self.seed_u)
        elif self.seed_x is not None:
            self.run(seed_x=self.seed_x)

    def run(self, seed_x: Union[list, np.ndarray] = None,
            seed_u: Union[list, np.ndarray] = None):
        """
        Runs FORM.

        :param seed_u: Either `seed_u` or `seed_x` must be provided.
         If `seed_u` is provided, it should be a point in the uncorrelated standard normal space of **U**.
         If `seed_x` is provided, it should be a point in the parameter space of **X**.
        :param seed_x: The initial starting point for the `Hasofer-Lind` algorithm.
         Either `seed_u` or `seed_x` must be provided.
         If `seed_u` is provided, it should be a point in the uncorrelated standard normal space of **U**.
         If `seed_x` is provided, it should be a point in the parameter space of **X**.
        """
        self.logger.info("UQpy: Running FORM...")
        if seed_u is None and seed_x is None:
            seed = np.zeros(self.dimension)
        elif seed_u is None and seed_x is not None:
            self.nataf_object.run(samples_x=seed_x.reshape(1, -1), jacobian=False)
            seed_z = self.nataf_object.samples_z
            seed = Decorrelate(seed_z, self.nataf_object.corr_z)
        elif seed_u is not None and seed_x is None:
            seed = np.squeeze(seed_u)
        else:
            raise ValueError("UQpy: Only one seed (seed_x or seed_u) must be provided")

        u_record = list()
        x_record = list()
        g_record = list()
        alpha_record = list()
        error_record = list()

        converged = False
        k = 0
        beta = np.zeros(shape=(self.n_iterations + 1,))
        u = np.zeros([self.n_iterations + 1, self.dimension])
        u[0, :] = seed
        g_record.append(0.0)
        dg_u_record = np.zeros([self.n_iterations + 1, self.dimension])

        while not converged and k < self.n_iterations:
            self.logger.info("Number of iteration: %i", k)
            # FORM always starts from the standard normal space
            if k == 0:
                if seed_x is not None:
                    x = seed_x
                else:
                    seed_z = Correlate(samples_u=seed.reshape(1, -1), corr_z=self.nataf_object.corr_z).samples_z
                    self.nataf_object.run(samples_z=seed_z.reshape(1, -1), jacobian=True)
                    x = self.nataf_object.samples_x
                    self.jacobian_zx = self.nataf_object.jxz
            else:
                z = Correlate(u[k, :].reshape(1, -1), self.nataf_object.corr_z).samples_z
                self.nataf_object.run(samples_z=z, jacobian=True)
                x = self.nataf_object.samples_x
                self.jacobian_zx = self.nataf_object.jxz

            self.x = x
            u_record.append(u)
            x_record.append(x)
            self.logger.info(
                "Design point Y: {0}\n".format(u[k, :])
                + "Design point X: {0}\n".format(self.x)
                + "Jacobian Jzx: {0}\n".format(self.jacobian_zx))

            # 2. evaluate Limit State Function and the gradient at point u_k and direction cosines
            dg_u, qoi, _ = self._derivatives(
                point_u=u[k, :],
                point_x=self.x,
                runmodel_object=self.runmodel_object,
                nataf_object=self.nataf_object,
                df_step=self.df_step,
                order="first")
            g_record.append(qoi)

            dg_u_record[k + 1, :] = dg_u
            norm_grad = np.linalg.norm(dg_u_record[k + 1, :])
            alpha = dg_u / norm_grad
            self.logger.info(
                "Directional cosines (alpha): {0}\n".format(alpha)
                + "Gradient (dg_y): {0}\n".format(dg_u_record[k + 1, :])
                + "norm dg_y:",
                norm_grad,)

            self.alpha = alpha.squeeze()
            alpha_record.append(self.alpha)
            beta[k] = -np.inner(u[k, :].T, self.alpha)
            beta[k + 1] = beta[k] + qoi / norm_grad
            self.logger.info(
                "Beta: {0}\n".format(beta[k])
                + "Pf: {0}".format(stats.norm.cdf(-beta[k])))

            u[k + 1, :] = -beta[k + 1] * self.alpha

            if (self.tol1 is not None) and (self.tol2 is not None) and (self.tol3 is not None):
                error1 = np.linalg.norm(u[k + 1, :] - u[k, :])
                error2 = np.linalg.norm(beta[k + 1] - beta[k])
                error3 = np.linalg.norm(dg_u_record[k + 1, :] - dg_u_record[k, :])
                error_record.append([error1, error2, error3])
                if error1 <= self.tol1 and error2 <= self.tol2 and error3 < self.tol3:
                    converged = True
                else:
                    k = k + 1

            if (self.tol1 is None) and (self.tol2 is None) and (self.tol3 is None):
                error1 = np.linalg.norm(u[k + 1, :] - u[k, :])
                error2 = np.linalg.norm(beta[k + 1] - beta[k])
                error3 = np.linalg.norm(dg_u_record[k + 1, :] - dg_u_record[k, :])
                error_record.append([error1, error2, error3])
                if error1 <= 1e-3 or error2 <= 1e-3 or error3 < 1e-3:
                    converged = True
                else:
                    k = k + 1

            elif (self.tol1 is not None) and (self.tol2 is None) and (self.tol3 is None):
                error1 = np.linalg.norm(u[k + 1, :] - u[k, :])
                error_record.append(error1)
                if error1 <= self.tol1:
                    converged = True
                else:
                    k = k + 1

            elif (
                (self.tol1 is None) and (self.tol2 is not None) and (self.tol3 is None)
            ):
                error2 = np.linalg.norm(beta[k + 1] - beta[k])
                error_record.append(error2)
                if error2 <= self.tol2:
                    converged = True
                else:
                    k = k + 1

            elif (self.tol1 is None) and (self.tol2 is None) and (self.tol3 is not None):
                error3 = np.linalg.norm(dg_u_record[k + 1, :] - dg_u_record[k, :])
                error_record.append(error3)
                if error3 < self.tol3:
                    converged = True
                else:
                    k = k + 1

            elif (self.tol1 is not None) and (self.tol2 is not None) and (self.tol3 is None):
                error1 = np.linalg.norm(u[k + 1, :] - u[k, :])
                error2 = np.linalg.norm(beta[k + 1] - beta[k])
                error_record.append([error1, error2])
                if error1 <= self.tol1 and error2 <= self.tol1:
                    converged = True
                else:
                    k = k + 1

            elif (self.tol1 is not None) and (self.tol2 is None) and (self.tol3 is not None):
                error1 = np.linalg.norm(u[k + 1, :] - u[k, :])
                error3 = np.linalg.norm(dg_u_record[k + 1, :] - dg_u_record[k, :])
                error_record.append([error1, error3])
                if error1 <= self.tol1 and error3 < self.tol3:
                    converged = True
                else:
                    k = k + 1


            elif (self.tol1 is None) and (self.tol2 is not None) and (self.tol3 is not None):
                error2 = np.linalg.norm(beta[k + 1] - beta[k])
                error3 = np.linalg.norm(dg_u_record[k + 1, :] - dg_u_record[k, :])
                error_record.append([error2, error3])
                if error2 <= self.tol2 and error3 < self.tol3:
                    converged = True
                else:
                    k = k + 1

            self.logger.info("Error: %s", error_record[-1])


        if k > self.n_iterations:
            self.logger.info("UQpy: Maximum number of iterations {0} was reached before convergence."
                             .format(self.n_iterations))
            self.error_record = error_record
            self.u_record = [u_record]
            self.x_record = [x_record]
            self.g_record = [g_record]
            self.dg_u_record = [dg_u_record[:k]]
            self.alpha_record = [alpha_record]
        else:
            if self.call is None:
                self.beta_record = [beta[:k]]
                self.error_record = error_record
                self.beta = [beta[k]]
                self.DesignPoint_U = [u[k, :]]
                self.DesignPoint_X = [np.squeeze(self.x)]
                self.failure_probability = [stats.norm.cdf(-self.beta[-1])]
                self.iterations = [k]
                self.u_record = [u_record[:k]]
                self.x_record = [x_record[:k]]
                self.g_record = [g_record]
                self.dg_u_record = [dg_u_record[:k]]
                self.alpha_record = [alpha_record]
            else:
                self.beta_record = self.beta_record + [beta[:k]]
                self.beta = self.beta + [beta[k]]
                self.error_record = self.error_record + error_record
                self.DesignPoint_U = self.DesignPoint_U + [u[k, :]]
                self.DesignPoint_X = self.DesignPoint_X + [np.squeeze(self.x)]
                self.failure_probability = self.failure_probability + [stats.norm.cdf(-beta[k])]
                self.iterations = self.iterations + [k]
                self.u_record = self.u_record + [u_record[:k]]
                self.x_record = self.x_record + [x_record[:k]]
                self.g_record = self.g_record + [g_record]
                self.dg_u_record = self.dg_u_record + [dg_u_record[:k]]
                self.alpha_record = self.alpha_record + [alpha_record]
            self.call = True

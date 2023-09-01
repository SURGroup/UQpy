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
        tolerance_u: Union[float, int, None] = 1e-3,
        tolerance_beta: Union[float, int, None] = 1e-3,
        tolerance_gradient: Union[float, int, None] = 1e-3,
    ):
        """
        A class perform the First Order reliability Method. Each time :meth:`run` is called the results are appended.
        This is a child class of the :class:`.TaylorSeries` class.

        :param distributions: Marginal probability distributions of each random variable. Must be an object of
         type :class:`.DistributionContinuous1D` or :class:`.JointIndependent`.
        :param runmodel_object: The computational model. It should be of type :class:`RunModel`.
        :param seed_x: The initial starting point in the parameter space **X** for the `Hasofer-Lind` algorithm.
         Either `seed_u` or `seed_x` must be provided.
         If either `seed_u` or `seed_x` is provided, then the :py:meth:`run` method will be executed automatically.
         Otherwise, the the :py:meth:`run` method must be executed by the user.
        :param seed_u: The initial starting point in the uncorrelated standard normal space **U** for the `Hasofer-Lind`
         algorithm. Either `seed_u` or `seed_x` must be provided. Default: :code:`seed_u = (0, 0, ..., 0)`
         If either `seed_u` or `seed_x` is provided, then the :py:meth:`run` method will be executed automatically.
         Otherwise, the the :py:meth:`run` method must be executed by the user.
        :param df_step: Finite difference step in standard normal space. Default: :math:`0.01`
        :param corr_x: Covariance matrix
         If `corr_x` is provided, it is the correlation matrix (:math:`\mathbf{C_X}`) of the random vector **X** .
        :param corr_z: Covariance matrix
         If `corr_z` is provided, it is the correlation matrix (:math:`\mathbf{C_Z}`) of the standard normal random
         vector **Z** . Default: `corr_z` is specified as the identity matrix.
        :param n_iterations: Maximum number of iterations for the `HLRF` algorithm. Default: :math:`100`
        :param tolerance_u: Convergence threshold for criterion :math:`||\mathbf{U}^{k} - \mathbf{U}^{k-1}||_2 \leq`
         :code:`tolerance_u` of the `HLRF` algorithm. Default: :math:`1.0e-3`
        :param tolerance_beta: Convergence threshold for criterion :math:`||\\beta_{HL}^{k}-\\beta_{HL}^{k-1}||_2 \leq`
         :code:`tolerance_beta` of the `HLRF` algorithm. Default: :math:`1.0e-3`
        :param tolerance_gradient: Convergence threshold for criterion
         :math:`||\\nabla G(\mathbf{U}^{k})- \\nabla G(\mathbf{U}^{k-1})||_2 \leq` :code:`tolerance_gradient`
         of the `HLRF` algorithm. Default: :math:`1.0e-3`

        By default, all three tolerances must be satisfied for convergence.
         Specifying a tolerance as :code:`None` will ignore that criteria.
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
        self.df_step = df_step
        self.dist_object = distributions
        self.n_iterations = n_iterations
        self.runmodel_object = runmodel_object
        self.seed_u = seed_u
        self.seed_x = seed_x
        self.tolerance_beta = tolerance_beta
        self.tolerance_gradient = tolerance_gradient
        self.tolerance_u = tolerance_u

        self.logger = logging.getLogger(__name__)

        # Initialize output
        self.alpha: float = np.nan
        """Direction cosine."""
        self.alpha_record: list = []
        """Record of the alpha (directional cosine)."""
        self.beta: list = []
        """Hasofer-Lind reliability index."""
        self.beta_record: list = []
        """Record of all Hasofer-Lind reliability index values."""
        self.design_point_u: list = []
        """Design point in the uncorrelated standard normal space U."""
        self.design_point_x: list = []
        """Design point in the parameter space X."""
        self.error_record: list = []
        """Record of the error defined by 
         criteria :math:`\\text{tolerance}_\\textbf{U}, \\text{tolerance}_\\beta, \\text{tolerance}_{\\nabla G(\\textbf{U})}`."""
        self.failure_probability: list = []
        """FORM probability of failure :math:`\\Phi(-\\beta)`."""
        self.g0 = None
        self.iterations: list = []
        """Number of model evaluations."""
        self.jacobian_zx = None
        """Jacobian of the transformation from correlated standard normal space to the parameter space."""
        self.state_function_record: list = []
        """Record of the performance function."""
        self.state_function_gradient_record: list = []
        """Record of the model’s gradient in the standard normal space."""
        self.u_record: list = []
        """Record of all iteration points in the standard normal space **U**."""
        self.x = None
        self.x_record: list = []
        """Record of all iteration points in the parameter space **X**."""

        if (seed_x is not None) and (seed_u is not None):
            raise ValueError('UQpy: Only one input (seed_x or seed_u) may be provided')
        if self.seed_u is not None:
            self.run(seed_u=self.seed_u)
        elif self.seed_x is not None:
            self.run(seed_x=self.seed_x)

    def run(self, seed_x: Union[list, np.ndarray] = None, seed_u: Union[list, np.ndarray] = None):
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
            seed = Decorrelate(seed_z, self.nataf_object.corr_z).samples_u
        elif seed_u is not None and seed_x is None:
            seed = np.squeeze(seed_u)
        else:
            raise ValueError("UQpy: Only one seed (seed_x or seed_u) must be provided")

        converged = False
        k = 0
        beta = np.zeros(shape=(self.n_iterations + 1,))
        u = np.zeros([self.n_iterations + 1, self.dimension])
        state_function_gradient_record = np.zeros([self.n_iterations + 1, self.dimension])
        u[0, :] = seed

        self.state_function_record.append(0.0)

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
            self.u_record.append(u)
            self.x_record.append(x)
            self.logger.info("Design point Y: {0}\n".format(u[k, :])
                             + "Design point X: {0}\n".format(self.x)
                             + "Jacobian Jzx: {0}\n".format(self.jacobian_zx))

            # 2. evaluate Limit State Function and the gradient at point u_k and direction cosines
            state_function_gradient, qoi, _ = self._derivatives(point_u=u[k, :],
                                                                point_x=self.x,
                                                                runmodel_object=self.runmodel_object,
                                                                nataf_object=self.nataf_object,
                                                                df_step=self.df_step,
                                                                order="first")
            self.state_function_record.append(qoi)
            state_function_gradient_record[k + 1, :] = state_function_gradient
            norm_of_state_function_gradient = np.linalg.norm(state_function_gradient)
            alpha = state_function_gradient / norm_of_state_function_gradient
            self.logger.info("Directional cosines (alpha): {0}\n".format(alpha)
                             + "State Function Gradient: {0}\n".format(state_function_gradient)
                             + "Norm of State Function Gradient: {0}\n".format(norm_of_state_function_gradient))
            self.alpha = alpha.squeeze()
            self.alpha_record.append(self.alpha)
            beta[k] = -np.inner(u[k, :].T, self.alpha)
            beta[k + 1] = beta[k] + qoi / norm_of_state_function_gradient
            self.logger.info("Beta: {0}\n".format(beta[k]) + "Pf: {0}".format(stats.norm.cdf(-beta[k])))

            u[k + 1, :] = -beta[k + 1] * self.alpha

            error_u = np.linalg.norm(u[k + 1, :] - u[k, :])
            error_beta = np.linalg.norm(beta[k + 1] - beta[k])
            error_gradient = np.linalg.norm(state_function_gradient - state_function_gradient_record[k, :])
            self.error_record.append([error_u, error_beta, error_gradient])

            converged_in_u = True if (self.tolerance_u is None) else (error_u <= self.tolerance_u)
            converged_in_beta = True if (self.tolerance_beta is None) else (error_beta <= self.tolerance_beta)
            converged_in_gradient = True if (self.tolerance_gradient is None) \
                else (error_gradient <= self.tolerance_gradient)
            converged = converged_in_u and converged_in_beta and converged_in_gradient
            if not converged:
                k += 1

            self.logger.info("Error: %s", self.error_record[-1])

        if k > self.n_iterations:
            self.logger.info("UQpy: Maximum number of iterations {0} was reached before convergence."
                             .format(self.n_iterations))
        else:
            self.design_point_u.append(u[k, :])
            self.design_point_x.append(np.squeeze(self.x))
            self.beta.append(beta[k])
            self.failure_probability.append(stats.norm.cdf(-beta[k]))
            self.state_function_gradient_record.append(state_function_gradient_record[:k])
            self.iterations.append(k)





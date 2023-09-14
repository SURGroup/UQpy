import logging
import warnings
import numpy as np
import scipy.stats as stats
from typing import Union
from beartype import beartype
from UQpy.distributions import *
from UQpy.transformations import *
from UQpy.run_model.RunModel import RunModel
from UQpy.utilities.ValidationTypes import PositiveInteger
from UQpy.reliability.taylor_series.baseclass.TaylorSeries import TaylorSeries

warnings.filterwarnings('ignore')


class InverseFORM(TaylorSeries):

    @beartype
    def __init__(
            self,
            distributions: Union[None, Distribution, list[Distribution]],
            runmodel_object: RunModel,
            p_fail: Union[None, float] = 0.05,
            beta: Union[None, float] = None,
            seed_x: Union[list, np.ndarray] = None,
            seed_u: Union[list, np.ndarray] = None,
            df_step: Union[int, float] = 0.01,
            corr_x: Union[list, np.ndarray] = None,
            corr_z: Union[list, np.ndarray] = None,
            max_iterations: PositiveInteger = 100,
            tolerance_u: Union[float, int, None] = 1e-3,
            tolerance_gradient: Union[float, int, None] = 1e-3,
    ):
        """Class to perform the Inverse First Order Reliability Method.

        Each time :meth:`run` is called the results are appended to the class attributes.
        By default, :code:`tolerance_u` and :code:`tolerance_gradient` must be satisfied for convergence.
        Specifying a tolerance as :code:`None` will ignore that criteria, but both cannot be :code:`None`.
        This is a child class of the :class:`TaylorSeries` class.

        :param distributions: Marginal probability distributions of each random variable. Must be of
         type :class:`.DistributionContinuous1D` or :class:`.JointIndependent`.
        :param runmodel_object: The computational model. Must be of type :class:`RunModel`.
        :param p_fail: Probability of failure defining the feasibility criteria as :math:`||u||=-\\Phi^{-1}(p_{fail})`.
         Default: :math:`0.05`. Set to :code:`None` to use :code:`beta` as the feasibility criteria.
         Only one of :code:`p_fail` or :code:`beta` may be provided.
        :param beta: Hasofer-Lind reliability index defining the feasibility criteria as :math:`||u|| = \\beta`.
         Default: :code:`None`.
         Only one of :code:`p_fail` or :code:`beta` may be provided.
        :param seed_x: Point in the parameter space :math:`\mathbf{X}` to start from.
         Only one of :code:`seed_x` or :code:`seed_u` may be provided.
         If either :code:`seed_u` or :code:`seed_x` is provided, then the :py:meth:`run` method will be executed automatically.
        :param seed_u: Point in the uncorrelated standard normal space :math:`\mathbf{U}` to start from.
         Only one of :code:`seed_x` or :code:`seed_u` may be provided.
         If either :code:`seed_u` or :code:`seed_x` is provided, then the :py:meth:`run` method will be executed automatically.
        :param df_step: Finite difference step in standard normal space. Default: :math:`0.01`
        :param corr_x: Correlation matrix :math:`\mathbf{C_X}` of the random vector :math:`\mathbf{X}`
        :param corr_z: Correlation matrix :math:`\mathbf{C_Z}` of the random vector :math:`\mathbf{Z}`
         Default: :code:`corr_z` is the identity matrix.
        :param max_iterations: Maximum number of iterations for the `HLRF` algorithm. Default: :math:`100`
        :param tolerance_u: Convergence threshold for criterion :math:`||\mathbf{U}_i - \mathbf{U}_{i-1}||_2 \leq`
         :code:`tolerance_u` of the `HLRF` algorithm.
         Default: :math:`1.0e-3`
        :param tolerance_gradient: Convergence threshold for criterion
         :math:`||\\nabla G(\mathbf{U}_i)- \\nabla G(\mathbf{U}_{i-1})||_2 \leq` :code:`tolerance_gradient`
         of the `HLRF` algorithm.
         Default: :math:`1.0e-3`
        """
        self.distributions = distributions
        self.runmodel_object = runmodel_object
        if (p_fail is not None) and (beta is not None):
            raise ValueError('UQpy: Exactly one input (p_fail or beta) must be provided')
        elif (p_fail is None) and (beta is None):
            raise ValueError('UQpy: Exactly one input (p_fail or beta) must be provided')
        elif p_fail is not None:
            self.p_fail = p_fail
            self.beta = -stats.norm.ppf(self.p_fail)
        elif beta is not None:
            self.p_fail = stats.norm.cdf(-1*beta)
            self.beta = beta
        self.seed_x = seed_x
        self.seed_u = seed_u
        self.df_step = df_step
        self.corr_x = corr_x
        self.corr_z = corr_z
        self.max_iterations = max_iterations
        self.tolerance_u = tolerance_u
        self.tolerance_gradient = tolerance_gradient
        if (self.tolerance_u is None) and (self.tolerance_gradient is None):
            raise ValueError('UQpy: At least one tolerance (tolerance_u or tolerance_gradient) must be provided')

        self.logger = logging.getLogger(__name__)
        self.nataf_object = Nataf(distributions=distributions, corr_z=corr_z, corr_x=corr_x)

        # Determine the number of dimensions as the number of random variables
        if isinstance(distributions, DistributionContinuous1D):
            self.dimension = 1
        elif isinstance(distributions, JointIndependent):
            self.dimension = len(distributions.marginals)
        elif isinstance(distributions, list):
            self.dimension = 0
            for i in range(len(distributions)):
                if isinstance(distributions[i], DistributionContinuous1D):
                    self.dimension += 1
                elif isinstance(distributions[i], JointIndependent):
                    self.dimension += len(distributions[i].marginals)

        # Initialize attributes
        self.alpha: float = np.nan
        """Normalized gradient vector in :math:`\\textbf{U}` space"""
        self.alpha_record: list = []
        """Record of :math:`\\alpha=\\frac{\\nabla G(u)}{||\\nabla G(u)||}`"""
        self.beta_record: list = []
        """Record of Hasofer-Lind reliability index that defines the feasibility criteria 
         :math:`||\\textbf{U}||=\\beta_{HL}`"""
        self.design_point_u: list = []
        """Design point in the standard normal space :math:`\\textbf{U}`"""
        self.design_point_x: list = []
        """Design point in the parameter space :math:`\\textbf{X}`"""
        self.error_record: list = []
        """Record of the final error defined by 
         :math:`error_u = ||u_{new} - u||` and :math:`error_{\\nabla G(u)} = || \\nabla G(u_{new}) - \\nabla G(u)||`"""
        self.iteration_record: list = []
        """Record of the number of iterations before algorithm termination"""
        self.failure_probability_record: list = []
        """Record of the probability of failure defined by :math:`p_{fail} = \\Phi(-\\beta_{HL})`"""
        self.state_function: list = []
        """State function :math:`G(u)` evaluated at each step in the optimization"""

        if (seed_x is not None) and (seed_u is not None):
            raise ValueError('UQpy: Only one input (seed_x or seed_u) may be provided')
        if self.seed_u is not None:
            self.run(seed_u=self.seed_u)
        elif self.seed_x is not None:
            self.run(seed_x=self.seed_x)

    def run(self, seed_x: Union[list, np.ndarray] = None, seed_u: Union[list, np.ndarray] = None):
        """Runs the inverse FORM algorithm.

        :param seed_x: Point in the parameter space :math:`\mathbf{X}` to start from.
         Only one of :code:`seed_x` or :code:`seed_u` may be provided.
         If neither is provided, the zero vector in :math:`\mathbf{U}` space is the seed.
        :param seed_u: Point in the uncorrelated standard normal space :math:`\mathbf{U}` to start from.
         Only one of :code:`seed_x` or :code:`seed_u` may be provided.
         If neither is provided, the zero vector in :math:`\mathbf{U}` space is the seed.
        """
        self.logger.info('UQpy: Running InverseFORM...')
        if (seed_x is not None) and (seed_u is not None):
            raise ValueError('UQpy: Only one input (seed_x or seed_u) may be provided')

        # Allocate u and the gradient of G(u) as arrays
        u = np.zeros([self.max_iterations + 1, self.dimension])
        state_function = np.full(self.max_iterations + 1, np.nan)
        state_function_gradient = np.zeros([self.max_iterations + 1, self.dimension])

        # Set initial seed. If both seed_x and seed_u are None, the initial seed is a vector of zeros in U space.
        if seed_u is not None:
            u[0, :] = seed_u
        elif seed_x is not None:
            self.nataf_object.run(samples_x=seed_x.reshape(1, -1), jacobian=False)
            seed_z = self.nataf_object.samples_z
            u[0, :] = Decorrelate(seed_z, self.nataf_object.corr_z).samples_u

        converged = False
        iteration = 0
        while (not converged) and (iteration < self.max_iterations):
            self.logger.info(f'Number of iteration: {iteration}')
            if iteration == 0:
                if seed_x is not None:
                    x = seed_x
                else:
                    seed_z = Correlate(samples_u=u[0, :].reshape(1, -1), corr_z=self.nataf_object.corr_z).samples_z
                    self.nataf_object.run(samples_z=seed_z.reshape(1, -1), jacobian=True)
                    x = self.nataf_object.samples_x
            else:
                z = Correlate(u[iteration, :].reshape(1, -1), self.nataf_object.corr_z).samples_z
                self.nataf_object.run(samples_z=z, jacobian=True)
                x = self.nataf_object.samples_x
            self.logger.info(f'Design Point U: {u[iteration, :]}\nDesign Point X: {x}\n')
            state_function_gradient[iteration + 1, :], qoi, _ = self._derivatives(point_u=u[iteration, :],
                                                                                  point_x=x,
                                                                                  runmodel_object=self.runmodel_object,
                                                                                  nataf_object=self.nataf_object,
                                                                                  df_step=self.df_step,
                                                                                  order='first')
            self.logger.info(f'State Function: {qoi}')
            state_function[iteration + 1] = qoi

            alpha = state_function_gradient[iteration + 1]
            alpha /= np.linalg.norm(state_function_gradient[iteration + 1])
            u[iteration + 1, :] = -alpha * self.beta

            error_u = np.linalg.norm(u[iteration + 1, :] - u[iteration, :])
            error_gradient = np.linalg.norm(state_function_gradient[iteration + 1, :]
                                            - state_function_gradient[iteration, :])

            converged_u = True if (self.tolerance_u is None) \
                else (error_u <= self.tolerance_u)
            converged_gradient = True if (self.tolerance_gradient is None) \
                else (error_gradient <= self.tolerance_gradient)
            converged = converged_u and converged_gradient

            if not converged:
                iteration += 1

        if iteration >= self.max_iterations:
            self.logger.info(f'UQpy: Maximum number of iterations {self.max_iterations} reached before convergence')
        self.alpha_record.append(alpha)
        self.beta_record.append(self.beta)
        self.design_point_u.append(u[iteration, :])
        self.design_point_x.append(x)
        self.error_record.append((error_u, error_gradient))
        self.iteration_record.append(iteration)
        self.failure_probability_record.append(self.p_fail)
        self.state_function.append(state_function)

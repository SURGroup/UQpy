import logging
from typing import Union

import numpy as np
import scipy.stats as stats
from beartype import beartype

from UQpy.run_model.RunModel import RunModel
from UQpy.distributions.baseclass import Distribution
from UQpy.utilities.ValidationTypes import PositiveInteger
from UQpy.reliability.taylor_series.FORM import FORM
from UQpy.reliability.taylor_series.baseclass.TaylorSeries import TaylorSeries


class SORM(TaylorSeries):

    @beartype
    def __init__(
            self,
            form_object: FORM,
            df_step: Union[float, int] = 0.01,
    ):
        """
        :class:`.SORM` is a child class of the :class:`.TaylorSeries` class. Input: The :class:`.SORM` class requires an
        object of type :class:`.FORM` as input.

        :param form_object: Object of type :class:`.FORM`
        :param df_step: Finite difference step in standard normal space. Default: :math:`0.01`
        """

        self.logger = logging.getLogger(__name__)
        self.df_step = df_step
        """Finite difference step in standard normal space."""

        self.failure_probability: float = None
        """FORM probability of failure :math:`\\Phi(-\\beta_{HL}) \\prod_{i=1}^{n-1} (1+\\beta_{HL}\\kappa_i)^{-\\frac{1}{2}}`."""
        self.beta_second_order: list = None
        """Hasofer-Lind reliability index using the SORM method."""
        self.call = None

        self.form_object = form_object
        self._run()

    @classmethod
    @beartype
    def build_from_first_order(
            cls,
            distributions: Union[None, Distribution, list[Distribution]],
            runmodel_object: RunModel,
            seed_x: Union[list, np.ndarray] = None,
            seed_u: Union[list, np.ndarray] = None,
            df_step: Union[int, float] = 0.01,
            corr_x: Union[list, np.ndarray] = None,
            corr_z: Union[list, np.ndarray] = None,
            n_iterations: PositiveInteger = 100,
            tolerance_u: Union[float, int] = None,
            tolerance_beta: Union[float, int] = None,
            tolerance_gradient: Union[float, int] = None,
    ):
        """
        :param distributions: Marginal probability distributions of each random variable. Must be an object of
         type :class:`.DistributionContinuous1D` or :class:`.JointIndependent`.
        :param runmodel_object: The computational model. It should be of type :class:`RunModel`.
        :param seed_x: The initial starting point for the `Hasofer-Lind` algorithm.
         Either `seed_u` or `seed_x` must be provided.
         If `seed_u` is provided, it should be a point in the uncorrelated standard normal space of **U**.
         If `seed_x` is provided, it should be a point in the parameter space of **X**.
        :param seed_u: Either `seed_u` or `seed_x` must be provided.
         If `seed_u` is provided, it should be a point in the uncorrelated standard normal space of **U**.
         If `seed_x` is provided, it should be a point in the parameter space of **X**.
        :param df_step: Finite difference step in standard normal space. Default: :math:`0.01`
        :param corr_x: Covariance matrix
         If `corr_x` is provided, it is the correlation matrix (:math:`\mathbf{C_X}`) of the random vector **X** .
        :param corr_z: Covariance matrix
         If `corr_z` is provided, it is the correlation matrix (:math:`\mathbf{C_Z}`) of the standard normal random
         vector **Z** . Default: `corr_z` is specified as the identity matrix.
         Default: `corr_z` is specified as the identity matrix.
        :param n_iterations: Maximum number of iterations for the `HLRF` algorithm. Default: :math:`100`
        :param tolerance_u: Convergence threshold for criterion :math:`||\mathbf{U}^{k} - \mathbf{U}^{k-1}||_2 \leq`
         :code:`tolerance_u` of the `HLRF` algorithm. Default: :math:`1.0e-3`
        :param tolerance_beta: Convergence threshold for criterion :math:`||\\beta_{HL}^{k}-\\beta_{HL}^{k-1}||_2 \leq`
         :code:`tolerance_beta` of the `HLRF` algorithm. Default: :math:`1.0e-3`
        :param tolerance_gradient: Convergence threshold for criterion
         :math:`||\\nabla G(\mathbf{U}^{k})- \\nabla G(\mathbf{U}^{k-1})||_2 \leq` :code:`tolerance_gradient`
         of the `HLRF` algorithm. Default: :math:`1.0e-3`

        Any number of tolerances can be provided. Only the provided tolerances will be considered for the convergence
        of the algorithm. In case none of the tolerances are provided then they are considered equal to :math:`1e-3` and
        all are checked for the convergence.
        """
        f = FORM(distributions, runmodel_object, seed_x, seed_u, df_step,
                 corr_x, corr_z, n_iterations, tolerance_u, tolerance_beta, tolerance_gradient)

        return cls(f, df_step)

    def _run(self):
        """
        Runs SORM
        """
        self.logger.info("UQpy: Calculating SORM correction...")
        beta = self.form_object.beta[-1]

        self.dimension = self.form_object.dimension
        model = self.form_object.runmodel_object
        state_function_gradient_record = self.form_object.state_function_gradient_record

        matrix_a = np.fliplr(np.eye(self.dimension))
        matrix_a[:, 0] = self.form_object.alpha

        q = np.zeros(shape=(self.dimension, self.dimension))
        q[:, 0] = matrix_a[:, 0] / np.linalg.norm(matrix_a[:, 0])

        for i in range(1, self.dimension):
            ai = matrix_a[:, i]
            for j in range(0, i):
                aj = matrix_a[:, j]
                t = ai.dot(aj)
                ai = ai - t * aj
            q[:, i] = ai / np.linalg.norm(ai)

        r1 = np.fliplr(q).T
        self.logger.info("UQpy: Calculating the hessian for SORM..")

        hessian_g = self._derivatives(point_u=self.form_object.design_point_u[-1],
                                      point_x=self.form_object.design_point_x[-1],
                                      runmodel_object=model,
                                      nataf_object=self.form_object.nataf_object,
                                      order="second",
                                      df_step=self.df_step,
                                      point_qoi=self.form_object.state_function_record[-1])

        matrix_b = np.dot(np.dot(r1, hessian_g), r1.T) / np.linalg.norm(state_function_gradient_record[-1])
        kappa = np.linalg.eig(matrix_b[: self.dimension - 1, : self.dimension - 1])
        if self.call is None:
            self.failure_probability = [stats.norm.cdf(-1 * beta) * np.prod(1 / (1 + beta * kappa[0]) ** 0.5)]
            self.beta_second_order = [-stats.norm.ppf(self.failure_probability)]
        else:
            self.failure_probability += [stats.norm.cdf(-1 * beta) * np.prod(1 / (1 + beta * kappa[0]) ** 0.5)]
            self.beta_second_order += [-stats.norm.ppf(self.failure_probability)]

        self.call = True

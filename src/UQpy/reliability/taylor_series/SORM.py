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
        """Design point in the parameter space **X**."""
        self.df_step = df_step
        """Record of the error defined by criteria `e1, e2, e3`."""

        self.failure_probability: float = None
        self.beta_second_order = None
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
            tol1: Union[float, int] = None,
            tol2: Union[float, int] = None,
            tol3: Union[float, int] = None,
    ):
        """
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
        f = FORM(distributions, runmodel_object, seed_x, seed_u, df_step,
                 corr_x, corr_z, n_iterations, tol1, tol2, tol3)

        return cls(f, df_step)

    def _run(self):
        """
        Runs SORM
        """
        self.logger.info("UQpy: Calculating SORM correction...")
        beta = self.form_object.beta[-1]

        self.dimension = self.form_object.dimension
        model = self.form_object.runmodel_object
        dg_u_record = self.form_object.dg_u_record

        matrix_a = np.fliplr(np.eye(self.dimension))
        matrix_a[:, 0] = self.form_object.alpha

        def normalize(v):
            return v / np.sqrt(v.dot(v))

        q = np.zeros(shape=(self.dimension, self.dimension))
        q[:, 0] = normalize(matrix_a[:, 0])

        for i in range(1, self.dimension):
            ai = matrix_a[:, i]
            for j in range(0, i):
                aj = matrix_a[:, j]
                t = ai.dot(aj)
                ai = ai - t * aj
            q[:, i] = normalize(ai)

        r1 = np.fliplr(q).T
        self.logger.info("UQpy: Calculating the hessian for SORM..")

        hessian_g = self._derivatives(point_u=self.form_object.DesignPoint_U[-1],
                                      point_x=self.form_object.DesignPoint_X[-1],
                                      runmodel_object=model,
                                      nataf_object=self.form_object.nataf_object,
                                      order="second",
                                      df_step=self.df_step,
                                      point_qoi=self.form_object.g_record[-1][-1])

        matrix_b = np.dot(np.dot(r1, hessian_g), r1.T) / np.linalg.norm(dg_u_record[-1])
        kappa = np.linalg.eig(matrix_b[: self.dimension - 1, : self.dimension - 1])
        if self.call is None:
            self.failure_probability = [stats.norm.cdf(-1 * beta) * np.prod(1 / (1 + beta * kappa[0]) ** 0.5)]
            self.beta_second_order = [-stats.norm.ppf(self.failure_probability)]
        else:
            self.failure_probability += [stats.norm.cdf(-1 * beta) * np.prod(1 / (1 + beta * kappa[0]) ** 0.5)]
            self.beta_second_order += [-stats.norm.ppf(self.failure_probability)]

        self.call = True

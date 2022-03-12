import logging
from typing import Union, Optional

import numpy as np
import scipy.stats as stats
from beartype import beartype

from UQpy.RunModel import RunModel
from UQpy.distributions.baseclass import Distribution
from UQpy.reliability.taylor_series.FORM import FORM
from UQpy.reliability.taylor_series.baseclass.TaylorSeries import TaylorSeries


class SORM(TaylorSeries):

    @beartype
    def __init__(
            self,
            form_object: FORM,
            runmodel_object: RunModel = None,
            distributions: Optional[Union[None, Distribution, list[Distribution]]] = None,
            seed_u: np.ndarray = None,
            seed_x: np.ndarray = None,
            df_step: Union[float, int] = 0.01,
            corr_x: np.ndarray = None,
            corr_z: np.ndarray = None,
            iterations_number: int = None,
            tol1: Union[float, int] = None,
            tol2: Union[float, int] = None,
            tol3: Union[float, int] = None,
    ):
        """
        :class:`.SORM` is a child class of the :class:`.TaylorSeries` class. Input: The :class:`.SORM` class requires an
        object of type :class:`.FORM` as input.

        :param form_object: Object of type :class:`.FORM`
        :param distributions: Marginal probability distributions of each random variable. Must be an object of
         type :class:`.DistributionContinuous1D` or :class:`.JointIndependent`.
        :param seed_u: seed_u: The initial starting point for the `Hasofer-Lind` algorithm.
         Either `seed_u` or `seed_x` must be provided.
         If `seed_u` is provided, it should be a point in the uncorrelated standard normal space of **U**.
         If `seed_x` is provided, it should be a point in the parameter space of **X**.
         Default: :code:`seed_u = (0, 0, ..., 0)`
        :param runmodel_object: The computational model. It should be of type :class:`RunModel`.
        :param df_step: Finite difference step in standard normal space. Default: :math:`0.01`
        :param corr_x: Covariance matrix
         If `corr_x` is provided, it is the correlation matrix (:math:`\mathbf{C_X}`) of the random vector **X** .
         If `corr_z` is provided, it is the correlation matrix (:math:`\mathbf{C_Z}`) of the standard normal random
         vector **Z** .
         Default: `corr_z` is specified as the identity matrix.
        :param iterations_number: Maximum number of iterations for the `HLRF` algorithm. Default: :math:`100`
        :param tol1: Convergence threshold for criterion `e1` of the `HLRF` algorithm. Default: :math:`1.0e-3`
        :param tol2: Convergence threshold for criterion `e2` of the `HLRF` algorithm. Default: :math:`1.0e-3`
        :param tol3: Convergence threshold for criterion `e3` of the  `HLRF` algorithm. Default: :math:`1.0e-3`
        """
        super().__init__(
            distributions,
            runmodel_object,
            form_object,
            corr_x,
            corr_z,
            seed_x,
            seed_u,
            iterations_number,
            tol1,
            tol2,
            tol3,
            df_step,
        )
        self.logger = logging.getLogger(__name__)
        self.beta_form: float = None
        """Hasofer-Lind reliability index."""
        self.DesignPoint_U: list = None
        """Design point in the uncorrelated standard normal space **U**."""
        self.DesignPoint_X: list = None
        """Design point in the parameter space **X**."""
        self.Pf_form = None
        self.form_iterations: int = None
        """Number of model evaluations."""
        self.u_record: list = None
        """Record of all iteration points in the standard normal space **U**."""
        self.x_record: list = None
        """Record of all iteration points in the parameter space **X**."""
        self.g_record: list = None
        """Record of the performance function."""
        self.dg_record = None
        self.beta_record: list = None
        """Record of all Hasofer-Lind reliability index values."""
        self.alpha_record: list = None
        """Record of the alpha (directional cosine)."""
        self.dg_u_record: list = None
        """Record of the model’s gradient in the standard normal space."""
        self.df_step = df_step
        self.error_record: float = None
        """Record of the error defined by criteria `e1, e2, e3`."""

        self.failure_probability = None
        self.beta_sorm = None
        self.call = None

        self.form_object = form_object
        self._run()

    def _run(self):
        """
        Runs SORM
        """
        self.logger.info("UQpy: Calculating SORM correction...")

        self.beta_form = self.form_object.beta_form[-1]
        self.nataf_object = self.form_object.nataf_object
        self.DesignPoint_U = self.form_object.DesignPoint_U[-1]
        self.DesignPoint_X = self.form_object.DesignPoint_X[-1]
        self.Pf_form = self.form_object.failure_probability
        self.form_iterations = self.form_object.form_iterations[-1]
        self.u_record = self.form_object.u_record
        self.x_record = self.form_object.x_record
        self.beta_record = self.form_object.beta_record
        self.g_record = self.form_object.g_record
        self.dg_u_record = self.form_object.dg_u_record
        self.alpha_record = self.form_object.alpha_record
        self.error_record = self.form_object.error_record

        self.dimension = self.form_object.dimension
        alpha = self.form_object.alpha
        model = self.form_object.runmodel_object
        dg_u_record = self.form_object.dg_u_record

        matrix_a = np.fliplr(np.eye(self.dimension))
        matrix_a[:, 0] = alpha

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

        hessian_g = self._derivatives(
            point_u=self.DesignPoint_U,
            point_x=self.DesignPoint_X,
            runmodel_object=model,
            nataf_object=self.nataf_object,
            order="second",
            df_step=self.df_step,
            point_qoi=self.g_record[-1][-1],
        )

        matrix_b = np.dot(np.dot(r1, hessian_g), r1.T) / np.linalg.norm(dg_u_record[-1])
        kappa = np.linalg.eig(matrix_b[: self.dimension - 1, : self.dimension - 1])
        if self.call is None:
            self.failure_probability = [
                stats.norm.cdf(-1 * self.beta_form)
                * np.prod(1 / (1 + self.beta_form * kappa[0]) ** 0.5)
            ]
            self.beta_sorm = [-stats.norm.ppf(self.failure_probability)]
        else:
            self.failure_probability = self.failure_probability + [
                stats.norm.cdf(-1 * self.beta_form)
                * np.prod(1 / (1 + self.beta_form * kappa[0]) ** 0.5)
            ]
            self.beta_sorm = self.beta_sorm + [
                -stats.norm.ppf(self.failure_probability)
            ]

        self.call = True

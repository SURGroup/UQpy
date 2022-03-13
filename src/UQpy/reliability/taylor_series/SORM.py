import logging
from typing import Union, Optional

import numpy as np
import scipy.stats as stats
from beartype import beartype

from UQpy.distributions.baseclass import DistributionContinuous1D
from UQpy.distributions.collection import JointIndependent
from UQpy.transformations.Nataf import Nataf
from UQpy.RunModel import RunModel
from UQpy.distributions.baseclass import Distribution
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

        self.failure_probability = None
        self.beta_second_order = None
        """Hasofer-Lind reliability index using the SORM method."""
        self.call = None

        self.form_object = form_object
        self._run()

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

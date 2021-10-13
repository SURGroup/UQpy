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
    """
    A class to perform the Second Order reliability Method. This class is used to correct the estimated FORM probability
    using second-order information.
    ``SORM`` is a child class of the ``taylor_series`` class.
    **Input:**
    The ``SORM`` class requires an object of type ``FORM`` as input.
    **Output/Returns:**
    The ``SORM`` class has the same outputs as the ``FORM`` class plus
    * **Pf_sorm** (`float`):
        Second-order probability of failure estimate.
    * **beta_sorm** (`float`):
        Second-order reliability index.
    **Methods:**
    """
    @beartype
    def __init__(self,
                 form_object: FORM,
                 distributions: Optional[Union[None, Distribution, list[Distribution]]] = None,
                 seed_u: np.ndarray = None,
                 seed_x: np.ndarray = None,
                 runmodel_object: RunModel = None,
                 df_step: Union[float, int] = 0.01,
                 corr_x: np.ndarray = None,
                 corr_z: np.ndarray = None,
                 iterations_number: int = None,
                 tol1: Union[float, int] = None,
                 tol2: Union[float, int] = None,
                 tol3: Union[float, int] = None):

        super().__init__(distributions, runmodel_object, form_object, corr_x, corr_z, seed_x, seed_u, iterations_number,
                         tol1, tol2, tol3, df_step)
        self.logger = logging.getLogger(__name__)
        self.beta_form = None
        self.DesignPoint_U = None
        self.DesignPoint_X = None
        self.Pf_form = None
        self.form_iterations = None
        self.u_record = None
        self.x_record = None
        self.g_record = None
        self.dg_record = None
        self.beta_record = None
        self.alpha_record = None
        self.dg_u_record = None
        self.df_step = df_step
        self.error_record = None

        self.failure_probability = None
        self.beta_sorm = None
        self.call = None

        self.form_object = form_object
        self._run()

    def _run(self):

        """
        Run SORM
        This is an instance method that runs SORM.
        """

        self.logger.info('UQpy: Calculating SORM correction...')

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
        self.logger.info('UQpy: Calculating the hessian for SORM..')

        hessian_g = self.derivatives(point_u=self.DesignPoint_U, point_x=self.DesignPoint_X,
                                     runmodel_object=model, nataf_object=self.nataf_object,
                                     order='second', df_step=self.df_step, point_qoi=self.g_record[-1][-1])

        matrix_b = np.dot(np.dot(r1, hessian_g), r1.T) / np.linalg.norm(dg_u_record[-1])
        kappa = np.linalg.eig(matrix_b[:self.dimension-1, :self.dimension-1])
        if self.call is None:
            self.failure_probability = [stats.norm.cdf(-1 * self.beta_form) *
                                        np.prod(1 / (1 + self.beta_form * kappa[0]) ** 0.5)]
            self.beta_sorm = [-stats.norm.ppf(self.failure_probability)]
        else:
            self.failure_probability = self.failure_probability + [stats.norm.cdf(-1 * self.beta_form) *
                                                                   np.prod(1 / (1 + self.beta_form * kappa[0]) ** 0.5)]
            self.beta_sorm = self.beta_sorm + [-stats.norm.ppf(self.failure_probability)]

        self.call = True

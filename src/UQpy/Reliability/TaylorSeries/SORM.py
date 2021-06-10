import numpy as np
import scipy.stats as stats

from UQpy.Reliability.TaylorSeries.FORM import FORM
from UQpy.Reliability.TaylorSeries.TaylorSeries import TaylorSeries


class SORM(TaylorSeries):
    """
    A class to perform the Second Order Reliability Method. This class is used to correct the estimated FORM probability
     using second-order information.
    ``SORM`` is a child class of the ``TaylorSeries`` class.
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

    def __init__(self, form_object, dist_object=None, seed_u=None, seed_x=None, runmodel_object=None, df_step=0.01,
                 corr_x=None, corr_z=None, n_iter=100, tol1=1.0e-3, tol2=1.0e-3, tol3=1.0e-3, verbose=False):

        super().__init__(dist_object, runmodel_object, form_object, seed_x, seed_u, df_step, corr_x, corr_z, n_iter,
                         tol1, tol2, tol3, verbose)

        #self.beta_form = None
        #self.DesignPoint_U = None
        #self.DesignPoint_X = None
        #self.Pf_form = None
        #self.form_iterations = None
        #self.u_record = None
        #self.x_record = None
        #self.g_record = None
        #self.dg_record = None
        #self.beta_record = None
        #self.alpha_record = None
        #self.dg_u_record = None
        #self.df_step = df_step
        #self.error_record = None

        #self.Pf_sorm = None
        #self.beta_sorm = None
        self.call = None
        if isinstance(form_object, FORM):
            self.form_object = form_object
            self._run()
        else:
            raise TypeError('UQpy: An object of type ``FORM`` is required to run SORM')

    def _run(self):

        """
        Run SORM
        This is an instance method that runs SORM.
        """

        if self.verbose:
            print('UQpy: Calculating SORM correction...')

        self.beta_form = self.form_object.beta_form[-1]
        self.nataf_object = self.form_object.nataf_object
        self.DesignPoint_U = self.form_object.DesignPoint_U[-1]
        self.DesignPoint_X = self.form_object.DesignPoint_X[-1]
        self.Pf_form = self.form_object.Pf_form
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
        if self.verbose:
            print('UQpy: Calculating the hessian for SORM..')
        print(self.DesignPoint_U, self.DesignPoint_X)
        hessian_g = self.derivatives(point_u=self.DesignPoint_U, point_x=self.DesignPoint_X,
                                     runmodel_object=model, nataf_object=self.nataf_object,
                                     order='second', df_step=self.df_step, point_qoi=self.g_record[-1][-1])

        matrix_b = np.dot(np.dot(r1, hessian_g), r1.T) / np.linalg.norm(dg_u_record[-1])
        kappa = np.linalg.eig(matrix_b[:self.dimension-1, :self.dimension-1])
        if self.call is None:
            self.Pf_sorm = [stats.norm.cdf(-1*self.beta_form) * np.prod(1 / (1 + self.beta_form * kappa[0]) ** 0.5)]
            self.beta_sorm = [-stats.norm.ppf(self.Pf_sorm)]
        else:
            self.Pf_sorm = self.Pf_sorm + [stats.norm.cdf(-1*self.beta_form) * np.prod(1 / (1 + self.beta_form *
                                                                                          kappa[0]) ** 0.5)]
            self.beta_sorm = self.beta_sorm + [-stats.norm.ppf(self.Pf_sorm)]

        self.call = True
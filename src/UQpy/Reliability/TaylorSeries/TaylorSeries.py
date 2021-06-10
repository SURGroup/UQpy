import numpy as np
from UQpy.Distributions import *
from UQpy.RunModel import RunModel
from UQpy.Transformations import *
from typing import Callable


########################################################################################################################
########################################################################################################################
#                                        First/Second order reliability method
########################################################################################################################
class TaylorSeries:
    """
    Perform First and Second Order Reliability (FORM/SORM) methods.
    This is the parent class to all Taylor series expansion algorithms.
    **Input:**
    * **dist_object** ((list of ) ``Distribution`` object(s)):
        Marginal probability distributions of each random variable. Must be an object of type
        ``DistributionContinuous1D`` or ``JointInd``.
    * **runmodel_object** (``RunModel`` object):
        The computational model. It should be of type ``RunModel`` (see ``RunModel`` class).
    * **form_object** (``FORM`` object):
        It should be of type ``FORM`` (see ``FORM`` class). Used to calculate SORM correction.
    * **corr_z** or **corr_x** (`ndarray`):
        Covariance matrix
        If `corr_x` is provided, it is the correlation matrix (:math:`\mathbf{C_X}`) of the random vector **X** .
        If `corr_z` is provided, it is the correlation matrix (:math:`\mathbf{C_Z}`) of the standard normal random
        vector **Z** .
         Default: `corr_z` is specified as the identity matrix.
    * **seed_u** or **seed_x** (`ndarray`):
        The initial starting point for the `Hasofer-Lind` algorithm.
        Either `seed_u` or `seed_x` must be provided.
        If `seed_u` is provided, it should be a point in the uncorrelated standard normal space of **U**.
        If `seed_x` is provided, it should be a point in the parameter space of **X**.
        Default: `seed_u = (0, 0, ..., 0)`
    * **tol1** (`float`):
         Convergence threshold for criterion `e1` of the `HLRF` algorithm.
         Default: 1.0e-3
    * **tol2** (`float`):
         Convergence threshold for criterion `e2` of the `HLRF` algorithm.
         Default: 1.0e-3
    * **tol3** (`float`):
         Convergence threshold for criterion `e3` of the  `HLRF` algorithm.
         Default: 1.0e-3
    * **n_iter** (`int`):
         Maximum number of iterations for the `HLRF` algorithm.
         Default: 100
    * **df_step** ('float'):
         Finite difference step in standard normal space.
         Default: 0.01 (see `derivatives` class)
    * **verbose** (`Boolean`):
        A boolean declaring whether to write text to the terminal.
    **Methods:**
    """

    def __init__(self, dist_object, runmodel_object, form_object=None, seed_x=None, seed_u=None, df_step=0.01,
                 corr_x=None, corr_z=None, n_iter=100, tol1=1.0e-3, tol2=1.0e-3, tol3=1.0e-3, verbose=False):

        if form_object is None:
            if isinstance(dist_object, list):
                self.dimension = len(dist_object)
                self.dimension = 0
                for i in range(len(dist_object)):
                    if isinstance(dist_object[i], DistributionContinuous1D):
                        self.dimension = self.dimension + 1
                    elif isinstance(dist_object[i], JointInd):
                        self.dimension = self.dimension + len(dist_object[i].marginals)
                    else:
                        raise TypeError('UQpy: A  ``DistributionContinuous1D`` or ``JointInd`` object must be '
                                        'provided.')
            else:
                if isinstance(dist_object, DistributionContinuous1D):
                    self.dimension = 1
                elif isinstance(dist_object, JointInd):
                    self.dimension = len(dist_object.marginals)
                else:
                    raise TypeError('UQpy: A  ``DistributionContinuous1D``  or ``JointInd`` object must be provided.')

            if not isinstance(runmodel_object, RunModel) and not isinstance(runmodel_object, Callable):
                raise TypeError('UQpy: A  ``RunModel`` or a Callable object must be '
                                'provided.')

            self.nataf_object = Nataf(dist_object=dist_object, corr_z=corr_z, corr_x=corr_x)

        else:
            pass

        self.corr_x = corr_x
        self.corr_z = corr_z
        self.dist_object = dist_object
        self.n_iter = n_iter
        self.runmodel_object = runmodel_object
        self.tol1 = tol1
        self.tol2 = tol2
        self.tol3 = tol3
        self.seed_u = seed_u
        self.seed_x = seed_x
        self.df_step = df_step
        self.verbose = verbose

    @staticmethod
    def derivatives(point_u=None, point_x=None, runmodel_object=None, nataf_object=None, order='first', point_qoi=None,
                    df_step=0.01, verbose=False):
        """
        A method to estimate the derivatives (1st-order, 2nd-order, mixed) of a function using a central difference
        scheme after transformation to the standard normal space.
        This is a static method of the ``FORM`` class.
        **Inputs:**
        * **point_u** (`ndarray`):
            Point in the uncorrelated standard normal space at which to evaluate the gradient with shape
            `samples.shape=(1, dimension)`.
            Either `point_u` or `point_x` must be specified. If `point_u` is specified, the derivatives are computed
            directly.
        * **point_x** (`ndarray`):
            Point in the parameter space at which to evaluate the model with shape
            `samples.shape=(1, dimension)`.
            Either `point_u` or `point_x` must be specified. If `point_x` is specified, the variable is transformed to
            standard normal using the ``Nataf`` transformation and derivatives are computed.
        * **runmodel_object** (``RunModel`` object):
            The computational model. It should be of type ``RunModel`` (see ``RunModel`` class).
        * **nataf_object** (``Nataf`` object):
            An object of the ``Nataf`` class (see ``Nataf`` class).
        * **order** (`str`):
            Order of the derivative. Available options: 'first', 'second', 'mixed'.
            Default: 'first'.
        * **point_qoi** (`float`):
            Value of the model evaluated at ``point_u``. Used only for second derivatives.
        * **df_step** (`float`):
            Finite difference step in standard normal space.
            Default: 0.01
        * **verbose** (Boolean):
            A boolean declaring whether to write text to the terminal.
        **Output/Returns:**
        * **du_dj** (`ndarray`):
            Vector of first-order derivatives (if order = 'first').
        * **d2u_dj** (`ndarray`):
            Vector of second-order derivatives (if order = 'second').
        * **d2u_dij** (`ndarray`):
            Vector of mixed derivatives (if order = 'mixed').
        """
        if nataf_object is None or not isinstance(nataf_object, Nataf):
            raise TypeError('UQpy: A  ``Nataf`` object must be '
                            'provided.')
        if runmodel_object is not None:
            if not isinstance(runmodel_object, RunModel) and not isinstance(runmodel_object, Callable):
                raise TypeError('UQpy: A  ``RunModel`` or a Callable object must be '
                                'provided.')
        if point_u is None and point_x is None:
            raise TypeError('UQpy: Either `point_u` or `point_x` must be specified.')

        list_of_samples = list()
        if point_x is not None:
            if order.lower() == 'first' or (order.lower() == 'second' and point_qoi is None):
                list_of_samples.append(point_x.reshape(1, -1))
        else:
            z_0 = Correlate(point_u.reshape(1, -1), nataf_object.corr_z).samples_z
            nataf_object.run(samples_z=z_0.reshape(1, -1), jacobian=False)
            temp_x_0 = nataf_object.samples_x
            x_0 = temp_x_0
            list_of_samples.append(x_0)

        for ii in range(point_u.shape[0]):
            y_i1_j = point_u.tolist()
            y_i1_j[ii] = y_i1_j[ii] + df_step

            z_i1_j = Correlate(np.array(y_i1_j).reshape(1, -1), nataf_object.corr_z).samples_z
            nataf_object.run(samples_z=z_i1_j.reshape(1, -1), jacobian=False)
            temp_x_i1_j = nataf_object.samples_x
            x_i1_j = temp_x_i1_j
            list_of_samples.append(x_i1_j)

            y_1i_j = point_u.tolist()
            y_1i_j[ii] = y_1i_j[ii] - df_step
            z_1i_j = Correlate(np.array(y_1i_j).reshape(1, -1), nataf_object.corr_z).samples_z
            nataf_object.run(samples_z=z_1i_j.reshape(1, -1), jacobian=False)
            temp_x_1i_j = nataf_object.samples_x
            x_1i_j = temp_x_1i_j
            list_of_samples.append(x_1i_j)

        array_of_samples = np.array(list_of_samples)
        array_of_samples = array_of_samples.reshape((len(array_of_samples), -1))

        if isinstance(runmodel_object, RunModel):
            runmodel_object.run(samples=array_of_samples, append_samples=False)
            y1 = runmodel_object.qoi_list
        elif isinstance(runmodel_object, Callable):
            y1 = runmodel_object(array_of_samples)

        if order.lower() == 'first':
            gradient = np.zeros(point_u.shape[0])

            for jj in range(point_u.shape[0]):
                qoi_plus = y1[2 * jj + 1]
                qoi_minus = y1[2 * jj + 2]
                gradient[jj] = ((qoi_plus - qoi_minus) / (2 * df_step))

            return gradient, y1[0], array_of_samples

        elif order.lower() == 'second':
            if verbose:
                print('UQpy: Calculating second order derivatives..')
            d2y_dj = np.zeros([point_u.shape[0]])

            if point_qoi is None:
                qoi = [y1[0]]
                output_list = y1
            else:
                qoi = [point_qoi]
                output_list = qoi + y1

            for jj in range(point_u.shape[0]):
                qoi_plus = output_list[2 * jj + 1]
                qoi_minus = output_list[2 * jj + 2]

                d2y_dj[jj] = ((qoi_minus - 2 * qoi[0] + qoi_plus) / (df_step ** 2))

            list_of_mixed_points = list()
            import itertools
            range_ = list(range(point_u.shape[0]))
            d2y_dij = np.zeros([int(point_u.shape[0] * (point_u.shape[0] - 1) / 2)])
            count = 0
            for i in itertools.combinations(range_, 2):
                y_i1_j1 = point_u.tolist()
                y_i1_1j = point_u.tolist()
                y_1i_j1 = point_u.tolist()
                y_1i_1j = point_u.tolist()

                y_i1_j1[i[0]] += df_step
                y_i1_j1[i[1]] += df_step

                y_i1_1j[i[0]] += df_step
                y_i1_1j[i[1]] -= df_step

                y_1i_j1[i[0]] -= df_step
                y_1i_j1[i[1]] += df_step

                y_1i_1j[i[0]] -= df_step
                y_1i_1j[i[1]] -= df_step

                z_i1_j1 = Correlate(np.array(y_i1_j1).reshape(1, -1), nataf_object.corr_z).samples_z
                nataf_object.run(samples_z=z_i1_j1.reshape(1, -1), jacobian=False)
                x_i1_j1 = nataf_object.samples_x
                list_of_mixed_points.append(x_i1_j1)

                z_i1_1j = Correlate(np.array(y_i1_1j).reshape(1, -1), nataf_object.corr_z).samples_z
                nataf_object.run(samples_z=z_i1_1j.reshape(1, -1), jacobian=False)
                x_i1_1j = nataf_object.samples_x
                list_of_mixed_points.append(x_i1_1j)

                z_1i_j1 = Correlate(np.array(y_1i_j1).reshape(1, -1), nataf_object.corr_z).samples_z
                nataf_object.run(samples_z=z_1i_j1.reshape(1, -1), jacobian=False)
                x_1i_j1 = nataf_object.samples_x
                list_of_mixed_points.append(x_1i_j1)

                z_1i_1j = Correlate(np.array(y_1i_1j).reshape(1, -1), nataf_object.corr_z).samples_z
                nataf_object.run(samples_z=z_1i_1j.reshape(1, -1), jacobian=False)
                x_1i_1j = nataf_object.samples_x
                list_of_mixed_points.append(x_1i_1j)

                count = count + 1

            array_of_mixed_points = np.array(list_of_mixed_points)
            array_of_mixed_points = array_of_mixed_points.reshape((len(array_of_mixed_points), -1))
            if isinstance(runmodel_object, RunModel):
                runmodel_object.run(samples=array_of_mixed_points, append_samples=False)
                y2 = runmodel_object.qoi_list
            elif isinstance(runmodel_object, Callable):
                y2 = runmodel_object(array_of_mixed_points)

            for j in range(count):
                qoi_0 = y2[4 * j]
                qoi_1 = y2[4 * j + 1]
                qoi_2 = y2[4 * j + 2]
                qoi_3 = y2[4 * j + 3]
                d2y_dij[j] = ((qoi_0 + qoi_3 - qoi_1 - qoi_2) / (4 * df_step * df_step))

            hessian = np.diag(d2y_dj)
            import itertools
            range_ = list(range(point_u.shape[0]))
            add_ = 0
            for i in itertools.combinations(range_, 2):
                hessian[i[0], i[1]] = d2y_dij[add_]
                hessian[i[1], i[0]] = hessian[i[0], i[1]]
                add_ += 1

            return hessian
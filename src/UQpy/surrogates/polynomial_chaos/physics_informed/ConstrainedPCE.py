import numpy as np

from sklearn import linear_model as regresion
from UQpy.surrogates.polynomial_chaos.physics_informed.Utilities import ortho_grid, derivative_basis
import copy
from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.Polynomials import Polynomials
from UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion import PolynomialChaosExpansion
from UQpy.surrogates.polynomial_chaos.physics_informed.PdePCE import PdePCE
from UQpy.surrogates.polynomial_chaos.physics_informed.PdeData import PdeData
from beartype import beartype

import logging


class ConstrainedPCE:
    @beartype
    def __init__(self, pde_data: PdeData,
                 pde_pce: PdePCE,
                 pce: PolynomialChaosExpansion):
        """
        Class for construction of physics-informed PCE using Karush-Kuhn-Tucker normal equations

        :param pde_data: an object of the :py:meth:`UQpy` :class:`PdeData` class
        :param pde_pce: an object of the :py:meth:`UQpy` :class:`PdePce` class
        :param pce:  initial pce, an object of the :py:meth:`UQpy` :class:`PolynomialChaosExpansion` class
        """

        self.pde_data = pde_data
        self.pde_pce = pde_pce
        self.initial_pce = pce

        self.dirichlet_bc = self.pde_data.extract_dirichlet()
        self.virtual_s = None
        self.virtual_x = None
        self.basis_extended = None
        self.y_extended = None
        self.kkt = None

    @beartype
    def estimate_error(self, pce: PolynomialChaosExpansion, standardized_sample: np.ndarray):
        """
        Estimate an error of the physics-informed PCE consisting of three parts: prediction errors in training data,
        errors in boundary conditions and violations of given PDE. Total error is sum of individual mean squared errors.

        :param pce:  initial pce, an object of the :py:meth:`UQpy` :class:`PolynomialChaosExpansion` class
        :param standardized_sample: virtual samples in standardized space for evaluation of errors in PDE

        :return: estimated total mean squared error
        """

        ypce = pce.predict(pce.experimental_design_input)

        err_data = (np.sum((pce.experimental_design_output - ypce) ** 2) / len(ypce))

        err_pde = np.abs(
            self.pde_pce.evaluate_pde(standardized_sample, pce, coefficients=pce.coefficients) - self.pde_pce.evaluate_pde_source(
                standardized_sample, multindex=pce.multi_index_set, coefficients=pce.coefficients))
        err_pde = np.mean(err_pde ** 2)
        err_bc = self.pde_pce.evaluate_boundary_conditions(len(standardized_sample), pce)
        err_bc = np.mean(err_bc ** 2)
        err_complete = (err_data + err_pde + err_bc)
        return err_complete

    @beartype
    def lar(self,
            n_error_points: int = 50,
            virtual_niters: bool = False,
            max_iterations: int = None,
            no_iterations: bool = False,
            min_basis_functions: int = 1,
            nvirtual: int = -1,
            target_error: float = 0):
        """
            Fit the sparse physics-informed PCE by Least Angle Regression from Karush-Kuhn-Tucker normal equations

            :param n_error_points: number of virtual samples used for estimation of an error
            :param virtual_niters: if True, minimum number of basis functions is equal to number of BCs
            :param max_iterations: maximum number of iterations for construction of LAR Path
            :param no_iterations: use all obtained basis functions in the first step, i.e. no iterations
            :param min_basis_functions: minimum number of basis functions for starting the iterative process
            :param nvirtual: set number of virtual points, -1 corresponds to the optimal number
            :param target_error: target error of iterative process
        """
        self.ols(calculate_coefficients=False, nvirtual=nvirtual)
        logger = logging.getLogger(__name__)
        pce = copy.deepcopy(self.initial_pce)

        if self.pde_pce.virtual_points_sampling is None:
            virtual_samples = ortho_grid(n_error_points, pce.inputs_number, -1.0, 1.0)
        else:
            virtual_x = self.pde_pce.virtual_points_sampling(n_error_points)
            virtual_samples = Polynomials.standardize_sample(virtual_x, pce.polynomial_basis.distributions)

        if max_iterations is None:
            max_iterations = self.pde_data.nconstraints + 200

        lar_path = regresion.lars_path(self.basis_extended, self.y_extended, max_iter=max_iterations)[1]

        steps = len(lar_path)

        logger.info('Cardinality of the identified sparse basis set: {}'.format(int(steps)))

        multindex = self.initial_pce.multi_index_set

        if steps < 3:
            raise Exception('LAR identified constant function! Check your data.')

        best_error = np.inf
        lar_basis = []
        lar_index = []
        lar_error = []

        if virtual_niters == True and min_basis_functions == 1:
            min_basis_functions = self.pde_data.nconstraints + 1

        if min_basis_functions > steps - 2 or no_iterations == True:
            min_basis_functions = steps - 3

        logger.info('Start of the iterative LAR algorithm ({} steps)'.format(steps - 2 - min_basis_functions))

        for i in range(min_basis_functions, steps - 2):

            mask = lar_path[:i]
            mask = np.concatenate([[0], mask])

            multindex_step = multindex[mask, :]
            basis_step = list(np.array(self.initial_pce.polynomial_basis.polynomials)[mask])

            lar_index.append(multindex_step)
            lar_basis.append(basis_step)

            pce.polynomial_basis.polynomials_number = len(basis_step)
            pce.polynomial_basis.polynomials = basis_step
            pce.multi_index_set = multindex_step
            pce.set_data(pce.experimental_design_input, pce.experimental_design_output)

            pce.coefficients = self.ols(pce, nvirtual=nvirtual)

            err = self.estimate_error(pce, virtual_samples)

            lar_error.append(err)

            if err < best_error:
                best_error = err
                best_basis = basis_step
                best_index = multindex_step

            if best_error < target_error:
                break

        logger.info('End of the iterative LAR algorithm')
        logger.info('Lowest obtained error {}'.format(best_error))

        if len(lar_error) > 1:
            pce.polynomial_basis.polynomials_number = len(best_basis)
            pce.polynomial_basis.polynomials = best_basis
            pce.multi_index_set = best_index
            pce.set_data(pce.experimental_design_input, pce.experimental_design_output)
            pce.coefficients = self.ols(pce, nvirtual=nvirtual)
            err = self.estimate_error(pce, virtual_samples)

        logger.info('Final PCE error {}'.format(err))

        self.lar_pce = pce
        self.lar_basis = best_basis
        self.lar_multindex = best_index
        self.lar_error = best_error
        self.lar_error_path = lar_error

    @beartype
    def ols(self, pce: PolynomialChaosExpansion = None,
            nvirtual: int = -1,
            calculate_coefficients: bool = True,
            return_coefficients: bool = True,
            n_error_points: int = 100):
        """
        Fit the sparse physics-informed PCE by ordinary least squares from Karush-Kuhn-Tucker normal equations

        :param pce: an object of the :class:`PolynomialChaosExpansion` class
        :param nvirtual: set number of virtual points, -1 corresponds to the optimal number
        :param calculate_coefficients: if True, estimate deterministic coefficients. Othewise, construct only KKT normal equations
         for sparse solvers
        :param return_coefficients: if True, return coefficients of pce
        :param n_error_points: number of virtual samples used for estimation of an error
        """
        if pce is None:
            pce = self.initial_pce
            return_coefficients = False

        multindex = pce.multi_index_set
        polynomialbasis = pce.design_matrix

        if multindex.ndim == 1:
            multindex = np.array(multindex).reshape(-1, 1)

        y = pce.experimental_design_output

        n_constraints = self.pde_data.nconstraints
        card_basis, nvar = multindex.shape

        if nvirtual == -1:
            nvirtual = card_basis - n_constraints

            if nvirtual < 0:
                nvirtual = 0

        if nvirtual == 0:
            b = np.zeros((n_constraints, 1))
            a = np.zeros((n_constraints, len(multindex)))
            kkt = np.zeros((card_basis + n_constraints, card_basis + n_constraints))
            right_vector = np.zeros((card_basis + n_constraints, 1))

        else:
            a = np.zeros((n_constraints + nvirtual, len(multindex)))
            b = np.zeros((n_constraints + nvirtual, 1))
            kkt = np.zeros((card_basis + n_constraints + nvirtual, card_basis + n_constraints + nvirtual))
            right_vector = np.zeros((card_basis + n_constraints + nvirtual, 1))

            if self.pde_pce.virtual_points_sampling is None:
                virtual_x = pce.polynomial_basis.distributions.rvs(nvirtual)
            else:
                virtual_x = self.pde_pce.virtual_points_sampling(nvirtual)

            virtual_s = Polynomials.standardize_sample(virtual_x, pce.polynomial_basis.distributions)

            self.virtual_s = virtual_s
            self.virtual_x = virtual_x

            b[-nvirtual:, 0] = self.pde_pce.evaluate_pde_source(self.virtual_s)

            a_pde = self.pde_pce.evaluate_pde(self.virtual_s, pce)

            a[n_constraints:, :] = a_pde

        if n_constraints > 0:
            a_const = []
            b_const = []
            for i in range(len(self.pde_data.der_orders)):

                if nvar > 1:
                    leadvariable = self.pde_data.bc_normals[i]
                else:
                    leadvariable = 0

                if self.pde_data.der_orders[i] > 0:
                    samples = self.pde_data.get_boundary_samples(self.pde_data.der_orders[i])
                    coord_x = samples[:, :-1]
                    bc_res = samples[:, -1]
                    coord_s = Polynomials.standardize_sample(coord_x,
                                                                              pce.polynomial_basis.distributions)
                    ac = derivative_basis(coord_s, pce, derivative_order=self.pde_data.der_orders[i],
                                                leading_variable=int(leadvariable))
                    a_const.append(ac)
                    b_const.append(bc_res.reshape(-1, 1))

            a_const = np.vstack(a_const).reshape(n_constraints, -1)
            b_const = np.vstack(b_const).reshape(n_constraints, -1)
            b[:n_constraints] = b_const
            a[:n_constraints, :] = a_const
        # Construct KKT system

        kkt[:card_basis, :card_basis] = np.dot(polynomialbasis.T, polynomialbasis)
        kkt[:card_basis, card_basis:] = a.T
        kkt[card_basis:, :card_basis] = a

        # get complete design matrix
        self.kkt = kkt
        self.basis_extended = np.r_[polynomialbasis, a]
        self.y_extended = np.r_[y, b.reshape(-1)]

        if calculate_coefficients:
            # Construct the right vector

            right_vector[:card_basis, 0] = np.dot(polynomialbasis.T, y).reshape(-1)
            right_vector[card_basis:, 0] = b.reshape(-1)

            # get the coefficients
            a_opt_c_lambda = np.linalg.pinv(kkt) @ right_vector
            a_opt_c = a_opt_c_lambda[:card_basis, 0]

            if not return_coefficients:
                self.initial_pce.coefficients = a_opt_c
                if self.pde_pce.virtual_points_sampling is None:
                    standardized_sample = ortho_grid(n_error_points, pce.inputs_number, -1.0, 1.0)
                else:
                    virtual_x = self.pde_pce.virtual_points_sampling(n_error_points)
                    standardized_sample = Polynomials.standardize_sample(virtual_x,
                                                                              pce.polynomial_basis.distributions)
                err = self.estimate_error(self.initial_pce, standardized_sample)
                self.ols_err = err
            else:
                return a_opt_c

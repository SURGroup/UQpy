import numpy as np
from scipy import special as sp
from scipy.special import legendre
from sklearn import linear_model as regresion
from UQpy.surrogates import *
from UQpy.distributions.collection import Uniform, Normal
import UQpy.surrogates.polynomial_chaos.physics_informed.Utilities as utils
import copy
from beartype import beartype

class ConstrainedPce:
    @beartype
    def __init__(self, pde_data, pde_pce, pce):
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

    def estimate_error(self, pce, verif_S):

        ypce = pce.predict(pce.experimental_design_input)

        err_data = (np.sum((pce.experimental_design_output - ypce) ** 2) / len(ypce))

        err_pde = np.abs(
            self.pde_pce.pde_eval(verif_S, pce, coefficients=pce.coefficients) - self.pde_pce.pderes_eval(verif_S,
                                                                                                          multindex=pce.multi_index_set,
                                                                                                          coefficients=pce.coefficients))
        err_pde = np.mean(err_pde ** 2)
        err_bc = self.pde_pce.bc_eval(len(verif_S), pce)
        err_bc = np.mean(err_bc ** 2)
        err_complete = (err_data + err_pde + err_bc)
        return err_complete

    def lar(self, n_PI=50, virtual_niters=False, max_niter=None, no_iter=False, minsize_basis=1, nvirtual=None,
            target_error=0):
        """
            Fit the sparse physics-informed PCE by Least Angle Regression from Karush-Kuhn-Tucker normal equations

            :param n_PI: number of virtual samples used for estimation of an error
            :param virtual_niters: if True, minimum number of basis functions is equal to number of BCs
            :param max_niter: maximum number of iterations for construction of LAR Path
            :param no_iter: use all obtained basis functions in the first step, i.e. no iterations
            :param minsize_basis: minimum number of basis functions for starting the iterative process
            :param nvirtual: set number of virtual points
            :param target_error: target error of iterative process
        """
        self.ols(calc_coeff=False, nvirtual=nvirtual)

        pce = copy.deepcopy(self.initial_pce)

        if self.pde_pce.virt_func is None:
            verif_S = utils.ortho_grid(n_PI, pce.inputs_number, -1, 1)
        else:
            virtual_x = self.pde_pce.virt_func(n_PI)
            verif_S = polynomial_chaos.Polynomials.standardize_sample(virtual_x, pce.polynomial_basis.distributions)

        if max_niter is None:
            max_niter = self.pde_data.nconst + 200

        lar_path = regresion.lars_path(self.basis_extended, self.y_extended, max_iter=max_niter)[1]

        steps = len(lar_path)

        print('Obtained Steps: ', steps)
        multindex = self.initial_pce.multi_index_set

        if steps < 3:
            raise Exception('LAR identified constant function! Check your data.')

        best_err = np.inf
        lar_basis = []
        lar_index = []
        lar_error = []

        if virtual_niters == True and minsize_basis == 1:
            minsize_basis = self.pde_data.nconst + 1

        if minsize_basis > steps - 2 or no_iter == True:
            minsize_basis = steps - 3

        for i in range(minsize_basis, steps - 2):
            print('\nStep No. ', i)
            mask = lar_path[:i]
            mask = np.concatenate([[0], mask])

            multindex_step = multindex[mask, :]
            basis_step = list(np.array(self.initial_pce.polynomial_basis.polynomials)[mask])

            lar_index.append(multindex_step)
            lar_basis.append(basis_step)

            pce.polynomial_basis.polynomials_number = len(basis_step)
            pce.polynomial_basis.polynomials = basis_step
            pce.multi_index_set = multindex_step
            pce.set_ed(pce.experimental_design_input, pce.experimental_design_output)

            pce.coefficients = self.ols(pce, nvirtual=nvirtual)

            err = self.estimate_error(pce, verif_S)
            print('Error: ', err)
            lar_error.append(err)

            if err < best_err:
                best_err = err
                best_basis = basis_step
                best_index = multindex_step

            if best_err < target_error:
                break

        print('Best error: ', best_err)

        if len(lar_error) > 1:
            pce.polynomial_basis.polynomials_number = len(best_basis)
            pce.polynomial_basis.polynomials = best_basis
            pce.multi_index_set = best_index
            pce.set_ed(pce.experimental_design_input, pce.experimental_design_output)
            pce.coefficients = self.ols(pce, nvirtual=nvirtual)
            err = self.estimate_error(pce, verif_S)
        print('Final error: ', err)
        self.lar_pce = pce
        self.lar_basis = best_basis
        self.lar_multindex = best_index
        self.lar_error = best_err
        self.lar_error_path = lar_error

    def ols(self, pce=None, nvirtual=None, calc_coeff=True, return_coeff=True, n_PI=100):
        """
        Fit the sparse physics-informed PCE by ordinary least squares from Karush-Kuhn-Tucker normal equations

        :param pce: an object of the :py:meth:`UQpy` :class:`PolynomialChaosExpansion` class
        :param nvirtual: set number of virtual points
        :param calc_coeff: if True, estimate deterministic coefficients. Othewise, construct only KKT normal equations
         for sparse solvers
        :param return_coeff: if True, return coefficients of pce
        :param n_PI: number of virtual samples used for estimation of an error
        """
        if pce is None:
            pce = self.initial_pce
            return_coeff = False

        multindex = pce.multi_index_set
        polynomialbasis = pce.design_matrix

        if multindex.ndim == 1:
            multindex = np.array(multindex).reshape(-1, 1)

        y = pce.experimental_design_output

        n_constraints = self.pde_data.nconst
        card_basis, nvar = multindex.shape

        if nvirtual is None:
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

            if self.pde_pce.virt_func is None:
                virtual_x = pce.polynomial_basis.distributions.rvs(nvirtual)
            else:
                virtual_x = self.pde_pce.virt_func(nvirtual)

            virtual_s = polynomial_chaos.Polynomials.standardize_sample(virtual_x, pce.polynomial_basis.distributions)

            self.virtual_s = virtual_s
            self.virtual_x = virtual_x

            b[-nvirtual:, 0] = self.pde_pce.pderes_eval(self.virtual_s)

            a_pde = self.pde_pce.pde_eval(self.virtual_s, pce)

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
                    samples = self.pde_data.get_bcsamples(self.pde_data.der_orders[i])
                    coord_x = samples[:, :-1]
                    bc_res = samples[:, -1]
                    coord_s = polynomial_chaos.Polynomials.standardize_sample(coord_x,
                                                                              pce.polynomial_basis.distributions)
                    ac = self.derivative_basis(coord_s, pce, der_order=self.pde_data.der_orders[i],
                                               variable=leadvariable)
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

        if calc_coeff:
            # Construct the right vector

            right_vector[:card_basis, 0] = np.dot(polynomialbasis.T, y).reshape(-1)
            right_vector[card_basis:, 0] = b.reshape(-1)

            # get the coefficients
            a_opt_c_lambda = np.linalg.pinv(kkt) @ right_vector
            a_opt_c = a_opt_c_lambda[:card_basis, 0]

            if not return_coeff:
                self.initial_pce.coefficients = a_opt_c
                if self.pde_pce.virt_func is None:
                    verif_S = utils.ortho_grid(n_PI, pce.inputs_number, -1, 1)
                else:
                    virtual_x = self.pde_pce.virt_func(n_PI)
                    verif_S = polynomial_chaos.Polynomials.standardize_sample(virtual_x,
                                                                              pce.polynomial_basis.distributions)
                err = self.estimate_error(self.initial_pce, verif_S)
                self.ols_err = err
            else:
                return a_opt_c

    #     def ols_iter(self,):

    def derivative_basis(self, s, pce=None, der_order=0, variable=None):

        if pce is None:
            multindex = self.initial_pce.multi_index_set
            joint_distribution = self.initial_pce.polynomial_basis.distributions
        else:
            multindex = pce.multi_index_set
            joint_distribution = pce.polynomial_basis.distributions

        card_basis, nvar = multindex.shape

        if nvar == 1:
            marginals = [joint_distribution]
        else:
            marginals = joint_distribution.marginals

        mask_herm = [type(marg) == Normal for marg in marginals]
        mask_lege = [type(marg) == Uniform for marg in marginals]

        if variable is not None:

            ns = multindex[:, variable]
            polysd = []

            if mask_lege[variable]:

                for n in ns:
                    polysd.append(legendre(n).deriv(der_order))

                prep_l_deriv = np.sqrt((2 * multindex[:, variable] + 1)).reshape(-1, 1)

                prep_deriv = []
                for poly in polysd:
                    prep_deriv.append(np.polyval(poly, s[:, variable]).reshape(-1, 1))

                prep_deriv = np.array(prep_deriv)

            mask_herm[variable] = False
            mask_lege[variable] = False

        prep_hermite = sp.eval_hermitenorm(multindex[:, mask_herm][:, np.newaxis, :], s[:, mask_herm])
        prep_legendre = sp.eval_legendre(multindex[:, mask_lege][:, np.newaxis, :], s[:, mask_lege])

        prep_fact = np.sqrt(sp.factorial(multindex[:, mask_herm]))
        prep = np.sqrt((2 * multindex[:, mask_lege] + 1))

        multivariate_basis = np.prod(prep_hermite / prep_fact[:, np.newaxis, :], axis=2).T
        multivariate_basis *= np.prod(prep_legendre * prep[:, np.newaxis, :], axis=2).T

        if variable is not None:
            multivariate_basis *= np.prod(prep_deriv * prep_l_deriv[:, np.newaxis, :], axis=2).T

        return multivariate_basis

import numpy as np
import copy
import UQpy.surrogates.polynomial_chaos.physics_informed.Utilities as utils
from beartype import beartype
from UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion import PolynomialChaosExpansion
from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.PolynomialBasis import PolynomialBasis
from UQpy.surrogates.polynomial_chaos.polynomials.baseclass.Polynomials import Polynomials

class ReducedPCE:
    @beartype
    def __init__(self, pce: PolynomialChaosExpansion, n_deterministic: int, deterministic_positions: list = None):
        """
        Class to create a reduced PCE filtering out deterministic input variables (e.g. geometry)

        :param pce: an object of the :py:meth:`UQpy` :class:`PolynomialChaosExpansion` class
        :param n_deterministic: number of deterministic input variables
        :param deterministic_positions: list of positions of deterministic random variables in input vector.
         If None, positions are assumed to be the first n_det variables.
        """

        if deterministic_positions is None:
            deterministic_positions = list(np.arange(n_deterministic))

        self.original_multindex = pce.multi_index_set
        self.original_beta = pce.coefficients
        self.original_P, self.nvar = self.original_multindex.shape
        self.original_pce = pce
        self.determ_pos = deterministic_positions

        #   select basis containing only deterministic variables
        determ_multi_index = np.zeros(self.original_multindex.shape)
        determ_selection_mask = [False] * self.nvar

        for i in deterministic_positions:
            determ_selection_mask[i] = True

        determ_multi_index[:, determ_selection_mask] = self.original_multindex[:, determ_selection_mask]

        self.determ_multi_index = determ_multi_index.astype(int)
        self.determ_basis = PolynomialBasis.construct_arbitrary_basis(self.nvar,
                                                                                       self.original_pce.polynomial_basis.distributions,
                                                                                       self.determ_multi_index)

        reduced_multi_mask = self.original_multindex > 0

        reduced_var_mask = [True] * self.nvar
        for i in deterministic_positions:
            reduced_var_mask[i] = False

        reduced_multi_mask = reduced_multi_mask * reduced_var_mask

        reduced_multi_index = np.zeros(self.original_multindex.shape) + (self.original_multindex * reduced_multi_mask)
        reduced_multi_index = reduced_multi_index[:, reduced_var_mask]

        self.reduced_positions = reduced_multi_mask.sum(axis=1) > 0
        reduced_multi_index = reduced_multi_index[self.reduced_positions, :]

        unique_basis, unique_positions, self.unique_indices = np.unique(reduced_multi_index, axis=0, return_index=True,
                                                                        return_inverse=True)

        P_unique, nrand = unique_basis.shape
        self.unique_basis = np.concatenate((np.zeros((1, nrand)), unique_basis), axis=0)

    @beartype
    def evaluate_coordinate(self, coordinates: np.ndarray, return_coefficients: bool = False):

        """
        Evaluate reduced PCE coefficients for given deterministic coordinates.

        :param coordinates: deterministic coordinates for evaluation of reduced PCE
        :param return_coefficients: if True, return a vector of deterministic coefficients, else return Mean and Variance
        :return: mean and variance, or a vector of deterministic coefficients if return_coeff
        """

        coord_x = np.zeros((1, self.nvar))
        coord_x[0, self.determ_pos] = coordinates

        determ_basis_eval = PolynomialBasis(self.nvar, len(self.determ_multi_index),
                                                             self.determ_multi_index, self.determ_basis,
                                                             self.original_pce.polynomial_basis.distributions).evaluate_basis(
            coord_x)
        return self._unique_coefficients(determ_basis_eval, return_coefficients)

    @beartype
    def derive_coordinate(self, coordinates: np.ndarray,
                          derivative_order: int,
                          leading_variable: int,
                          derivative_multiplier: float = 1,
                          return_coefficients: bool = False):

        """
        Evaluate derivative of reduced PCE coefficients for given deterministic coordinates.

        :param coordinates: deterministic coordinates for evaluation of reduced PCE
        :param derivative_order: derivation order of reduced PCE
        :param leading_variable: leading variable for derivation
        :param derivative_multiplier: multiplier reflecting different sizes of the original physical and transformed spaces
        :param return_coefficients: if True, return a vector of deterministic coefficients, else return Mean and Variance
        :return: mean and variance, or a vector of deterministic coefficients if return_coeff
        """

        coord_x = np.zeros((1, self.nvar))
        coord_x[0, self.determ_pos] = coordinates
        coord_s = Polynomials.standardize_sample(coord_x,
                                                                  self.original_pce.polynomial_basis.distributions)

        determ_multi_index = np.zeros(self.original_multindex.shape)
        determ_selection_mask = np.arange(self.nvar) == self.determ_pos

        determ_multi_index[:, determ_selection_mask] = self.original_multindex[:, determ_selection_mask]
        determ_multi_index = determ_multi_index.astype(int)

        pce_deriv = copy.deepcopy(self.original_pce)
        pce_deriv.multi_index_set = determ_multi_index
        determ_basis_eval = utils.derivative_basis(coord_s, pce_deriv, derivative_order=derivative_order,
                                                   leading_variable=leading_variable) * (
                derivative_multiplier)

        return self._unique_coefficients(determ_basis_eval, return_coefficients)

    @beartype
    def variance_contributions(self, unique_beta: np.ndarray):

        """
        Get first order conditional variances from coefficients of reduced PCE evaluated in the specified deterministic
        physical coordinates

        :param unique_beta: vector of reduced PCE coefficients
        :return: first order conditional variances associated to each input random variable
        """

        variance = np.sum(unique_beta[1:] ** 2)
        multi_index_set = self.unique_basis
        terms, inputs_number = multi_index_set.shape
        variances = np.zeros(inputs_number)
        # take all multi-indices except 0-index
        idx_no_0 = np.delete(multi_index_set, 0, axis=0)
        for nn in range(inputs_number):
            # remove nn-th column
            idx_no_0_nn = np.delete(idx_no_0, nn, axis=1)
            # we want the rows with all indices (except nn) equal to zero
            sum_idx_rows = np.sum(idx_no_0_nn, axis=1)
            zero_rows = np.asarray(np.where(sum_idx_rows == 0)).flatten() + 1
            variance_contribution = np.sum(unique_beta[zero_rows] ** 2, axis=0)

            variances[nn] = variance_contribution

        return variances

    def _unique_coefficients(self, determ_basis_eval, return_coefficients):
        determ_beta = np.transpose(determ_basis_eval * self.original_beta)

        reduced_beta = determ_beta[self.reduced_positions]
        complement_beta = determ_beta[~self.reduced_positions]

        unique_beta = []
        for ind in np.unique(self.unique_indices):
            sum_beta = np.sum(reduced_beta[self.unique_indices == ind, 0])
            unique_beta.append(sum_beta)

        unique_beta = np.array([0] + unique_beta)
        unique_beta[0] = unique_beta[0] + np.sum(complement_beta)

        if not return_coefficients:
            mean = unique_beta[0]
            var = np.sum(unique_beta[1:] ** 2)
            return mean, var
        else:
            return unique_beta

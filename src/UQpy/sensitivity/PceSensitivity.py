from typing import Annotated

import numpy as np
from beartype.vale import Is

from UQpy.surrogates.polynomial_chaos import PolynomialChaosExpansion

FittedPce = Annotated[PolynomialChaosExpansion, Is[lambda pce: pce.coefficients is not None]]


class PceSensitivity:

    def __init__(self, pce_object: FittedPce):
        """
        Compute Sobol sensitivity indices based on a PCE surrogate approximation of the QoI.

        :param pce_object: Polynomial Chaos Expansion surrogate.
        """
        self.pce_object = pce_object
        self.first_order_indices = None
        """PCE estimates for the first order Sobol indices"""
        self.total_order_indices = None
        """PCE estimates for the total order Sobol indices"""
        self.generalized_first_order_indices = None
        """PCE estimates of generalized first order Sobol indices, which characterize
        the sensitivity of a vector-valued quantity of interest on the random
        inputs."""
        self.generalized_total_order_indices = None
        """PCE estimates of generalized total order Sobol indices, which characterize
        the sensitivity of a vector-valued quantity of interest on the random
        inputs."""

    def run(self):
        """
        This method calculates and saves as attributes first and total order as well as generalized sensitivity indices.
        """
        self.calculate_first_order_indices()
        self.calculate_total_order_indices()
        self.calculate_generalized_first_order_indices()
        self.calculate_generalized_total_order_indices()

    def calculate_first_order_indices(self) -> np.ndarray:
        """
        PCE estimates for the first order Sobol indices.

        :return: First order Sobol indices.
        """
        outputs_number = np.shape(self.pce_object.coefficients)[1]
        variance = self.pce_object.get_moments()[1]
        inputs_number = self.pce_object.inputs_number
        multi_index_set = self.pce_object.multi_index_set

        first_order_indices = np.zeros([inputs_number, outputs_number])
        # take all multi-indices except 0-index
        idx_no_0 = np.delete(multi_index_set, 0, axis=0)
        for nn in range(inputs_number):
            # remove nn-th column
            idx_no_0_nn = np.delete(idx_no_0, nn, axis=1)
            # we want the rows with all indices (except nn) equal to zero
            sum_idx_rows = np.sum(idx_no_0_nn, axis=1)
            zero_rows = np.asarray(np.where(sum_idx_rows == 0)).flatten() + 1
            variance_contribution = np.sum(self.pce_object.coefficients[zero_rows, :] ** 2, axis=0)
            first_order_indices[nn, :] = variance_contribution / variance
        self.first_order_indices = first_order_indices
        return first_order_indices

    def calculate_total_order_indices(self) -> np.ndarray:
        """
        PCE estimates for the total order Sobol indices.

        :return: Total order Sobol indices.
        """
        outputs_number = np.shape(self.pce_object.coefficients)[1]
        variance = self.pce_object.get_moments()[1]
        inputs_number = self.pce_object.inputs_number
        multi_index_set = self.pce_object.multi_index_set

        total_order_indices = np.zeros([inputs_number, outputs_number])
        for nn in range(inputs_number):
            # we want all multi-indices where the nn-th index is NOT zero
            idx_column_nn = np.array(multi_index_set)[:, nn]
            nn_rows = np.asarray(np.where(idx_column_nn != 0)).flatten()
            variance_contribution = np.sum(self.pce_object.coefficients[nn_rows, :] ** 2, axis=0)
            total_order_indices[nn, :] = variance_contribution / variance
        self.total_order_indices = total_order_indices
        return total_order_indices

    def calculate_generalized_first_order_indices(self) -> np.ndarray:
        """
        PCE estimates of generalized first order Sobol indices, which characterize
        the sensitivity of a vector-valued quantity of interest on the random
        inputs.

        :return: Generalized first order Sobol indices.
        """
        inputs_number = self.pce_object.inputs_number
        if inputs_number == 1:
            raise ValueError('Not applicable for scalar model outputs.')

        variance = self.pce_object.get_moments()[1]
        first_order_indices = self.calculate_first_order_indices()
        variance_contributions = first_order_indices * variance
        total_variance = np.sum(variance)
        total_variance_contribution_per_input = np.sum(variance_contributions, axis=1)
        generalized_first_order_indices = total_variance_contribution_per_input / total_variance
        self.generalized_first_order_indices = generalized_first_order_indices
        return generalized_first_order_indices

    def calculate_generalized_total_order_indices(self):
        """
        PCE estimates of generalized total order Sobol indices, which characterize
        the sensitivity of a vector-valued quantity of interest on the random
        inputs.

        :return: Generalized total order Sobol indices.
        """
        inputs_number = self.pce_object.inputs_number

        if inputs_number == 1:
            raise ValueError('Not applicable for scalar model outputs.')

        variance = self.pce_object.get_moments()[1]
        total_order_indices = self.calculate_total_order_indices()
        variance_contributions = total_order_indices * variance
        total_variance = np.sum(variance)
        total_variance_contribution_per_input = np.sum(variance_contributions, axis=1)
        generalized_total_order_indices = total_variance_contribution_per_input / total_variance
        self.generalized_total_order_indices = generalized_total_order_indices
        return generalized_total_order_indices

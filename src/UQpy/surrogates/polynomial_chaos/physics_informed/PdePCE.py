import numpy as np
from UQpy.surrogates.polynomial_chaos.physics_informed.PdeData import PdeData
from typing import Callable
from UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion import PolynomialChaosExpansion

class PdePCE:
    def __init__(self, pde_data: PdeData,
                 pde_basis: Callable,
                 pde_source: Callable = None,
                 boundary_conditions_evaluate: Callable = None,
                 boundary_conditions_sampling: Callable = None,
                 virtual_points_sampling: Callable = None,
                 nonlinear: bool = False):
        """
        Class containing information about PDE needed for physics-informed PCE

        :param pde_data: an object of the :code:`UQpy` :class:`.PdeData` class
        :param pde_basis: pde defined in basis functions
        :param pde_source: source term of pde
        :param boundary_conditions_evaluate: evaluation of boundary conditions for estimation of an error
        :param boundary_conditions_sampling: function for sampling of boundary conditions
        :param virtual_points_sampling: function for sampling of virtual samples
        :param nonlinear: if True, prescribed pde is non-linear
        """

        self.pde_data = pde_data
        self.pde_basis = pde_basis
        self.pde_source = pde_source
        self.boundary_conditions_sampling = boundary_conditions_sampling
        self.boundary_conditions_evaluate = boundary_conditions_evaluate
        self.virtual_points_sampling = virtual_points_sampling
        self.nonlinear = nonlinear

    def evaluate_pde(self, standardized_sample: np.ndarray,
                     pce: PolynomialChaosExpansion,
                     coefficients: np.ndarray = None):

        pde_basis = self.pde_basis(standardized_sample, pce)

        if coefficients is not None:
            return np.sum(pde_basis * np.array(coefficients).T, axis=1)
        else:
            return pde_basis

    def evaluate_boundary_conditions(self,
                                     nsim: np.ndarray,
                                     pce: PolynomialChaosExpansion):
        return self.boundary_conditions_evaluate(nsim, pce)

    def evaluate_pde_source(self, standardized_sample: np.ndarray, multindex: np.ndarray = None,
                            coefficients: np.ndarray = None):
        if self.pde_source is not None:
            if self.nonlinear:
                return self.pde_source(standardized_sample, multindex, coefficients)
            else:
                return self.pde_source(standardized_sample)
        else:
            return 0


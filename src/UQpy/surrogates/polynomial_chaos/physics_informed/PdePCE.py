import numpy as np


class PdePCE:
    def __init__(self, pde_data,
                 pde_functions,
                 pde_source=None,
                 boundary_conditions=None,
                 boundary_condition_function=None,
                 virtual_points_function=None,
                 nonlinear=False):
        """
        Class containing information about PDE needed for physics-informed PCE

        :param pde_data: an object of the :code:`UQpy` :class:`.PdeData` class
        :param pde_functions: pde defined in basis functions
        :param pde_source: source term of pde
        :param boundary_conditions: evaluation of boundary conditions for estimation of an error
        :param boundary_condition_function: function for sampling of boundary conditions
        :param virtual_points_function: function for sampling of virtual samples
        :param nonlinear: if True, prescribed pde is non-linear
        """

        self.pde_data = pde_data
        self.pde_function = pde_functions
        self.pde_source = pde_source
        self.boundary_condition_function = boundary_condition_function
        self.boundary_conditions = boundary_conditions
        self.virtual_functions = virtual_points_function
        self.nonlinear = nonlinear

    def evaluate_pde(self, s,
                     pce,
                     coefficients=None):

        pde_basis = self.pde_function(s, pce)

        if coefficients is not None:
            return np.sum(pde_basis * np.array(coefficients).T, axis=1)
        else:
            return pde_basis

    def evaluate_boundary_conditions(self,
                                     nsim,
                                     pce):
        return self.boundary_conditions(nsim, pce)

    def evaluate_pde_source(self, s, multindex=None, coefficients=None):
        if self.pde_source is not None:
            if self.nonlinear:
                return self.pde_source(s, multindex, coefficients)
            else:
                return self.pde_source(s)
        else:
            return 0

    def evaluate_boundary_condition_function(self, s, multindex=None, coefficients=None):
        if self.boundary_condition_function is not None:
            if coefficients is not None:
                bc_pce = self.boundary_condition_function(s, multindex, coefficients)
            else:
                bc_pce = self.boundary_condition_function(s)
            return bc_pce
        else:
            return 0

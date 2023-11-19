import numpy as np

class PdePCE:
    def __init__(self, pde_data, pde_func, pde_res=None, bc_res=None, bc_func=None, virt_func=None, nonlinear=False):
        """
        Class containing information about PDE needed for physics-informed PCE

        :param pde_data: an object of the :py:meth:`UQpy` :class:`PdeData` class
        :param pde_func: pde defined in basis functions
        :param pde_res: source term of pde
        :param bc_res: evaluation of boundary conditions for estimation of an error
        :param bc_func: function for sampling of boundary conditions
        :param virt_func: function for sampling of virtual samples
        :param nonlinear: if True, prescribed pde is non-linear
        """

        self.pde_data = pde_data
        self.pde = pde_func
        self.pde_res = pde_res
        self.bc_func = bc_func
        self.bc_res = bc_res
        self.virt_func = virt_func
        self.nonlinear = nonlinear

    def pde_eval(self, s, pce, coefficients=None):

        pde_basis = self.pde(s, pce)

        if coefficients is not None:
            return np.sum(pde_basis * np.array(coefficients).T, axis=1)
        else:
            return pde_basis

    def bc_eval(self, nsim, pce):
        return self.bc_res(nsim, pce)

    def pderes_eval(self, s, multindex=None, coefficients=None):

        if self.pde_res is not None:
            if self.nonlinear:
                return self.pde_res(s, multindex, coefficients)
            else:
                return self.pde_res(s)
        else:
            return 0

    def bcfunc_eval(self, s, multindex=None, coefficients=None):
        if self.bc_func is not None:
            if coefficients is not None:
                bc_pce = self.bc_func(s, multindex, coefficients)
            else:
                bc_pce = self.bc_func(s)
            return bc_pce
        else:
            return 0
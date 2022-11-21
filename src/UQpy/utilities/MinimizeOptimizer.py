from scipy.optimize import minimize
import logging


class MinimizeOptimizer:

    def __init__(self, method: str = 'l-bfgs-b', bounds=None):
        # super().__init__(bounds)
        self._bounds = None
        self.logger = logging.getLogger(__name__)
        self.optimization = minimize
        self.method = method
        self.save_bounds(bounds)
        self.constraints = {}

    def save_bounds(self, bounds):
        if self.method.lower() in ['nelder-mead', 'l-bfgs-b', 'tnc', 'slsqp', 'powell', 'trust-constr']:
            self._bounds = bounds
        else:
            self.logger.warning("The selected optimizer method does not support bounds and thus will be ignored.")

    def optimize(self, function, initial_guess, args=(), jac=False):
        if self.constraints:
            return minimize(function, initial_guess, args=args,
                            method=self.method, bounds=self._bounds,
                            constraints=self.constraints, jac=jac,
                            options={'disp': False, 'maxiter': 10000, 'catol': 0.002})
        else:
            return minimize(function, initial_guess, args=args,
                            method=self.method, bounds=self._bounds, jac=jac,
                            options={'disp': False, 'maxiter': 10000, 'catol': 0.002})

    def apply_constraints(self, constraints):
        if self.method.lower() in ['cobyla', 'slsqp', 'trust-constr']:
            self.constraints = constraints
        else:
            self.logger.warning("The selected optimizer method does not support constraints and thus will be ignored.")

    def update_bounds(self, bounds):
        self.save_bounds(bounds)

    def supports_jacobian(self):
        return self.method.lower() in ['cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'slsqp', 'dogleg', 'trust-ncg',
                                       'trust-krylov', 'trust-exact', 'trust-constr']

from scipy.optimize import fmin_cobyla
import logging


class FminCobyla:

    def __init__(self):
        # super().__init__(None)
        self.logger = logging.getLogger(__name__)
        self.optimization = fmin_cobyla
        self.method = 'cobyla'

        self.constraints = {}
        self.arguments = {}

    def optimize(self, function, initial_guess, args=(), jac=False):
        if self.constraints:
            return fmin_cobyla(function, initial_guess, cons=self.constraints,
                               args=args, consargs=self.arguments, rhobeg=1.0, rhoend=0.0001, maxfun=1000,
                               disp=None, catol=0.0002)
        else:
            return fmin_cobyla(function, initial_guess, args=args,
                               rhobeg=1.0, rhoend=0.0001, maxfun=1000,
                               disp=None, catol=0.0002)

    def apply_constraints(self, constraints):
        self.constraints = constraints

    def apply_constraints_argument(self, arguments):
        self.arguments = arguments

    def supports_jacobian(self):
        return self.method.lower() in ['cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'slsqp', 'dogleg', 'trust-ncg',
                                       'trust-krylov', 'trust-exact', 'trust-constr']

    def update_bounds(self, bounds):
        pass

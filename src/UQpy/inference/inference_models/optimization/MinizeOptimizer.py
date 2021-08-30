from scipy.optimize import minimize

from UQpy.inference.inference_models.optimization.Optimizer import Optimizer


class MinimizeOptimizer(Optimizer):

    def __init__(self, method=None, bounds=None):
        self.optimization = minimize
        self.method = method
        self.bounds = bounds

    def optimize(self, function, initial_guess):
        return minimize(function, initial_guess, method=self.method, bounds=self.bounds)
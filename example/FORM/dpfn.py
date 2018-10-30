import numpy as np
import numdifftools as nd


class RunPythonModel:

    def __init__(self, samples=None, dimension=None):

        self.dimension = dimension
        self.samples = samples
        self.Grad = self.grad(samples)
        self.Hessian = self.hessian(samples)

    def fun(self, x):

        beta = 3
        return beta*np.sqrt(self.dimension) - (x[0] + x[1])

    def grad(self, x):

        return nd.Gradient(self.fun)(x)

    def hessian(self, x):

        return nd.Hessian(self.fun)(x)

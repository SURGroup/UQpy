"""

Rosenbrock Distribution Auxiliary File
======================================================================

"""
from UQpy.distributions import DistributionND
import numpy as np

class Rosenbrock(DistributionND):
    def __init__(self, p=20.):
        super().__init__(p=p)

    def pdf(self, x):
        return np.exp(-(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2) / self.parameters['p'])

    def log_pdf(self, x):
        return -(100*(x[:, 1]-x[:, 0]**2)**2+(1-x[:, 0])**2)/self.parameters['p']

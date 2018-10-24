
import numpy as np


class RunPythonModel:

    def __init__(self, samples=None, dimension=None):

        self.dimension = dimension
        self.samples = samples

        self.QOI = [0]*self.samples.shape[0]

        domain = np.linspace(0, 10, 50)

        for i in range(self.samples.shape[0]):
            self.QOI[i] = self.samples[i, 0]*domain+self.samples[i, 1]*domain**2+self.samples[i, 2]*domain**3
import numpy as np
import sys


class RunPythonModel:

    def __init__(self, samples=None, dimension=None):

        self.samples = samples
        self.dimension = dimension
        self.QOI = [0]*self.samples.shape[0]

        beta = 3
        for i in range(self.samples.shape[0]):
            self.QOI[i] = beta*np.sqrt(self.dimension) - np.sum(self.samples[i])
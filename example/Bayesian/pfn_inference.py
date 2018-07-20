
import numpy as np


class RunPythonModel:

    def __init__(self, samples=None, dimension=None):

        self.dimension = dimension
        self.samples = samples

        x_1 = 36 * 0.992
        x_2 = 1.05 * 0.75
        x_4 = 29000 * 0.9875
        x_5 = 1.0 * 0.35
        x_6 = 1.0 * 5.25

        self.QOI = [0]*self.samples.shape[0]

        for i in range(self.samples.shape[0]):
            x_3 = self.samples[i] + 34
            self.QOI[i] = (2.1/np.sqrt(x_1**2 * x_3/(x_2**2*x_4))-0.9/(np.sqrt(x_1**2*x_3/(x_2**2*x_4))**2)) * \
                          (1-0.75*x_5/np.sqrt(x_1**2*x_3/(x_2**2*x_4)))*(1-2*x_6*x_2/x_1)
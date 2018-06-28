import numpy as np


class RunPythonModel:

    def __init__(self, samples=None, dimension=None):

        self.samples = samples
        self.dimension = dimension

        der_ = np.zeros(2)
        der_[0] = -0.6071
        der_[1] = -0.8071

        self.QOI = [-0.6071*self.samples[0, 0] - 0.8071*self.samples[0, 1] + 2.5, der_]


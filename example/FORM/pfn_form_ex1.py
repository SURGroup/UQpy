import numpy as np


class RunPythonModel:

    def __init__(self, samples=None, dimension=None):

        self.samples = samples
        self.dimension = dimension

        beta = 3.5
        self.QOI = [-np.sum(self.samples) / np.sqrt(self.dimension) + beta, np.tile(-1 /
                          np.sqrt(self.dimension), [self.dimension, 1])]

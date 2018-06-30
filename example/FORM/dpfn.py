import numpy as np


class RunPythonModel:

    def __init__(self, samples=None, dimension=None):

        self.samples = samples
        self.dimension = dimension

        self.QOI = np.tile(-1/np.sqrt(self.dimension), [self.dimension,1])

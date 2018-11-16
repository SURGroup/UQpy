
import numpy as np


class RunPythonModel:

    def __init__(self, samples=None, dimension=None):

        self.dimension = dimension
        self.samples = samples

        domain = np.linspace(0, 10, 50)

        self.qoi = np.array([sample[0]*domain for sample in self.samples])
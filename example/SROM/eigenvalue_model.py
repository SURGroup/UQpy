import numpy as np


class RunPythonModel:

    def __init__(self, samples=None, dimension=None):

        self.samples = samples
        self.dimension = dimension
        self.qoi = np.zeros_like(self.samples)
        for i in range(self.samples.shape[0]):
            p = np.array([[self.samples[i, 0]+self.samples[i, 1], -self.samples[i, 1], 0], 
                          [-self.samples[i, 1], self.samples[i, 1]+self.samples[i, 2], -self.samples[i, 2]], 
                          [0, -self.samples[i, 2], self.samples[i, 2]]])
            w, v = np.linalg.eig(p)
            self.qoi[i, :] = w


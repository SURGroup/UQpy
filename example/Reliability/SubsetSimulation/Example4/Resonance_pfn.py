import numpy as np


class RunPythonModel:

    def __init__(self, samples=None):

        self.samples = samples
        self.qoi = [0]*self.samples.shape[0]

        self.omega = 6.
        self.epsilon = 0.0001
        
        for i in range(self.samples.shape[0]):
            add = self.samples[i][1] - self.samples[i][0]*(self.omega+self.epsilon)**2
            diff = self.samples[i][0]*(self.omega-self.epsilon)**2 - self.samples[i][1]
            self.qoi[i] = np.maximum(add, diff)
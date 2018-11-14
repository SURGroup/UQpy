import numpy as np
import sys


#class RunPythonModel:

#    def __init__(self, samples=None, dimension=None):

#        self.samples = samples
#        self.dimension = dimension
#        self.QOI = [0]*self.samples.shape[0]

#        beta = 3
#        for i in range(self.samples.shape[0]):
#            self.QOI[i] = beta * np.sqrt(self.dimension) - np.sum(self.samples[i])
            
            
class RunPythonModel:

    def __init__(self, samples=None, dimension=None):

        self.samples = samples
        self.dimension = dimension
        self.QOI = [0]*self.samples.shape[0]

        E = 5
        epsilon = 1
        
        for i in range(self.samples.shape[0]):
            x0 = 0
            x1 = 0
            x0 = self.samples[i,0]+5
            x1 = self.samples[i,1]*10+125
            g1 = x1-x0*(E-epsilon)**2
            g2 = x0*(E+epsilon)**2-x1
            #print(self.samples[i,0])
            #print(self.samples[i,1])
            #print(x0)
            #print(x1)
            #print(g1)
            #print(g2)
            self.QOI[i] = np.maximum(g1,g2)
            #print(self.QOI[i])
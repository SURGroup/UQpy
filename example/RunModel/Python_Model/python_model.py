import numpy as np
import sys


class RunPythonModel:

    def __init__(self, samples=None, dimension=None):

        self.samples = samples
        self.dimension = dimension
        self.QOI = [0]*self.samples.shape[0]

        for i in range(self.samples.shape[0]):
            self.QOI[i] = np.sum(self.samples[i])

    # index = sys.argv[1]
    # filename = 'modelInput_{0}.txt'.format(int(index))
    # x = np.loadtxt(filename, dtype=np.float32)
    #
    # p = np.sqrt(abs(np.sum(x)))
    #
    # with open('solution_{0}.txt'.format(int(index)), 'w') as f:
    #     f.write('{} \n'.format(p))

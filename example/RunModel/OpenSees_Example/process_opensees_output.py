import numpy as np


def read_output(index):
    x = np.loadtxt("./OutputFiles/node20001_%d.out" % index)
    return x[-1, 1]


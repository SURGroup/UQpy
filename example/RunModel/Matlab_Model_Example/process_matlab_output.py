import numpy as np


def read_output(index):
    x = np.loadtxt("./OutputFiles/oupt_%d.out" % index)
    return x


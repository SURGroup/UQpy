import numpy as np


def read_output(index):
    x = np.load("./OutputFiles/oupt_%d.npy" % index)
    return x

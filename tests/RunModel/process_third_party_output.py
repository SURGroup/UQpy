import numpy as np
import os

def read_output(index):
    x = np.load("./OutputFiles/oupt_%d.npy" % index)
    return x

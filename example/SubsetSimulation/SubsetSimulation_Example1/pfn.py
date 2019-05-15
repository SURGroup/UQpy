import numpy as np
import sys


def RunPythonModel(samples):

    qoi = list()
    beta = 3.0902
    for i in range(samples.shape[0]):
        qoi.append(-1/np.sqrt(2) * (samples[0, 0] + samples[0, 1]) + beta)
    return qoi
            
            
import numpy as np


def example1(samples=None):
    g = np.zeros(samples.shape[0])
    for i in range(samples.shape[0]):
        R = samples[i, 0]
        S = samples[i, 1]
        g[i] = R - S
    return g
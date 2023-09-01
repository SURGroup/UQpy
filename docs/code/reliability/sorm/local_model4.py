import numpy as np


def example4(samples=None):
    g = np.zeros(samples.shape[0])
    for i in range(samples.shape[0]):
        g[i] = samples[i, 0] * samples[i, 1] - 80
    return g
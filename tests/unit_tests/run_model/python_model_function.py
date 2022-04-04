import numpy as np


def sum_rvs(samples=None):
    x = np.sum(samples, axis=1)
    return x

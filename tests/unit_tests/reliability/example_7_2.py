import numpy as np


def performance_function(samples=None):
    """Performance function from Chapter 7 Example 7.2 from Du 2005"""
    elastic_modulus = 30e6
    length = 100
    width = 2
    height = 4
    d_0 = 3

    g = np.zeros(samples.shape[0])
    for i in range(samples.shape[0]):
        x = (samples[i, 0] / width**2) ** 2
        y = (samples[i, 1] / height**2) ** 2
        d = ((4 * length**3) / (elastic_modulus * width * height)) * np.sqrt(x + y)
        g[i] = d_0 - d
    return g

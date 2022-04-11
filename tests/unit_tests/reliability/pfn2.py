import numpy as np


def model_j(samples):
    d0 = 3
    e = 30000000
    l = 100
    w = 2
    t = 4
    return d0 - 4 * l ** 3 / (e * w * t) * np.sqrt((samples[0, 1] / t ** 2) ** 2 + (samples[0, 0] / w ** 2) ** 2)
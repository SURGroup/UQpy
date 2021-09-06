import numpy as np


def model_i(samples):
    resistance = samples[0, 0]
    stress = samples[0, 1]
    return resistance - stress

def model_j(samples):
    d0 = 3
    e = 30000000
    l = 100
    w = 2
    t = 4
    return d0 - 4 * l ** 3 / (e * w * t) * np.sqrt((samples[0, 1] / t ** 2) ** 2 + (samples[0, 0] / w ** 2) ** 2)

def model_k(samples):
    return samples[0, 0] * samples[0, 1] - 80

def example1(samples=None):
    g = np.zeros(samples.shape[0])
    for i in range(samples.shape[0]):
        R = samples[i, 0]
        S = samples[i, 1]
        g[i] = R - S
    return g


def example2(samples=None):
    import numpy as np
    d = 2
    beta = 3.0902
    g = np.zeros(samples.shape[0])
    for i in range(samples.shape[0]):
        g[i] = -1 / np.sqrt(d) * (samples[i, 0] + samples[i, 1]) + beta
    return g


def example3(samples=None):
    g = np.zeros(samples.shape[0])
    for i in range(samples.shape[0]):
        g[i] = 6.2 * samples[i, 0] - samples[i, 1] * samples[i, 2] ** 2
    return g


def example4(samples=None):
    g = np.zeros(samples.shape[0])
    for i in range(samples.shape[0]):
        g[i] = samples[i, 0] * samples[i, 1] - 80
    return g

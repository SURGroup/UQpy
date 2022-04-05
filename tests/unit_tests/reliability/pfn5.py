import numpy as np


def Example1(samples=None):

    x = np.zeros(samples.shape[0])

    omega = 6.
    epsilon = 0.0001

    for i in range(samples.shape[0]):
        add = samples[i][1] - samples[i][0]*(omega+epsilon)**2
        diff = samples[i][0]*(omega-epsilon)**2 - samples[i][1]
        x[i] = np.maximum(add, diff)

    return x
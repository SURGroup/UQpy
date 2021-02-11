import numpy as np
import sys


class RunPythonModel:

    def __init__(self, samples=None, dimension=None):

        self.samples = samples
        self.dimension = dimension
        self.qoi = [0]*self.samples.shape[0]

        P = 750
        for i in range(self.samples.shape[0]):
            self.qoi[i] = 1800-np.maximum(self.samples[i,0],self.samples[i,1])-2*np.sqrt(2)/3*P


def gfun_sensitivity(samples, a_values):
    gi_xi = [(np.abs(4. * Xi - 2) + ai) / (1. + ai) for Xi, ai in zip(np.array(samples).T, a_values)]
    gfun = np.prod(np.array(gi_xi), axis=0)
    return list(gfun)


def fun2_sensitivity(samples):
    fun_vals = 0.01 * samples[:, 0] + 1. * samples[:, 1] + 0.4 * samples[:, 2] ** 2 + samples[:, 3] * samples[:, 4]
    return list(fun_vals)


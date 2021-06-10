import numpy as np


def gfun_sensitivity(samples, a_values):
    gi_xi = [(np.abs(4. * Xi - 2) + ai) / (1. + ai) for Xi, ai in zip(np.array(samples).T, a_values)]
    gfun = np.prod(np.array(gi_xi), axis=0)
    return list(gfun)

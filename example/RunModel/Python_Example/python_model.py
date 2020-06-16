import numpy as np


def sum_rvs(samples=None):
    x = np.sum(samples, axis=1)
    return x


def sum_rvs_vec(samples=None):
    x = np.sum(samples, axis=2)
    return x


class SumRVs:
    def __init__(self, samples=None):

        self.qoi = np.sum(samples, axis=1)

class SumRVsVec:
    def __init__(self, samples=None):

        self.qoi = np.sum(samples, axis=2)


def det_rvs(samples=None):

    x = samples[:][0] * np.linalg.det(samples[:][1])
    return x


def det_rvs_par(samples=None):
    x = samples[0][0] * np.linalg.det(samples[0][1])
    return x


class DetRVs:
    def __init__(self, samples=None):

        self.qoi = samples[0][0] * np.linalg.det(samples[0][1])


def det_rvs_fixed(samples=None, coeff=None):

    x = coeff * np.linalg.det(samples[:])
    return x

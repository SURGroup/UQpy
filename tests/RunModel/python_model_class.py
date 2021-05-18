import numpy as np


class SumRVs:
    def __init__(self, samples=None):

        self.qoi = np.sum(samples, axis=1)

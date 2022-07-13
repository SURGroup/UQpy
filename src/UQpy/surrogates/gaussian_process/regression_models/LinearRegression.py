import numpy as np
from UQpy.surrogates.gaussian_process.regression_models.baseclass.Regression import Regression


class LinearRegression(Regression):
    def r(self, s):
        s = np.atleast_2d(s)
        fx = np.concatenate((np.ones([np.size(s, 0), 1]), s), 1)
        # jf_b = np.zeros([np.size(s, 0), np.size(s, 1), np.size(s, 1)])
        # np.einsum("jii->ji", jf_b)[:] = 1
        # jf = np.concatenate((np.zeros([np.size(s, 0), np.size(s, 1), 1]), jf_b), 2)
        return fx

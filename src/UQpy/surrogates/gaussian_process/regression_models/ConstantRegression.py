import numpy as np
from UQpy.surrogates.gaussian_process.regression_models.baseclass.Regression import Regression


class ConstantRegression(Regression):
    def r(self, s):
        s = np.atleast_2d(s)
        # jf = np.zeros([np.size(s, 0), np.size(s, 1), 1])
        return np.ones([np.size(s, 0), 1])

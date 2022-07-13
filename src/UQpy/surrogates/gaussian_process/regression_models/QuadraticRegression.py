import numpy as np
from UQpy.surrogates.gaussian_process.regression_models.baseclass.Regression import Regression


class QuadraticRegression(Regression):
    def r(self, s):
        s = np.atleast_2d(s)
        fx = np.zeros(
            [np.size(s, 0), int((np.size(s, 1) + 1) * (np.size(s, 1) + 2) / 2)]
        )
        # jf = np.zeros(
        #     [
        #         np.size(s, 0),
        #         np.size(s, 1),
        #         int((np.size(s, 1) + 1) * (np.size(s, 1) + 2) / 2),
        #     ]
        # )
        for i in range(np.size(s, 0)):
            temp = np.hstack((1, s[i, :]))
            for j in range(np.size(s, 1)):
                temp = np.hstack((temp, s[i, j] * s[i, j::]))
            fx[i, :] = temp
            # definie H matrix
            # h_ = 0
            # for j in range(np.size(s, 1)):
            #     tmp_ = s[i, j] * np.eye(np.size(s, 1))
            #     t1 = np.zeros([np.size(s, 1), np.size(s, 1)])
            #     t1[j, :] = s[i, :]
            #     tmp = tmp_ + t1
            #     if j == 0:
            #         h_ = tmp[:, j::]
            #     else:
            #         h_ = np.hstack((h_, tmp[:, j::]))
            # jf[i, :, :] = np.hstack(
            #     (np.zeros([np.size(s, 1), 1]), np.eye(np.size(s, 1)), h_)
            # )
        return fx

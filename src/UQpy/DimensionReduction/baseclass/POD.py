import numpy as np


########################################################################################################################
########################################################################################################################
#                                                     POD                                                              #
########################################################################################################################
########################################################################################################################


class POD:
    """
    Performs Direct and Snapshot Proper Orthogonal Decomposition (POD) as well as Higher-order Singular Value
    Decomposition (HOSVD) for dimension reduction of datasets.

    **Input:**

    * **input_sol** (`ndarray`) or (`list`):
        Second order tensor or list containing the solution snapshots. Third dimension or length of list corresponds
        to the number of snapshots.

    * **verbose** (`Boolean`):
        A boolean declaring whether to write text to the terminal.

    **Methods:**
    """

    def __init__(self, input_sol, verbose=True, **kwargs):

        self.input_sol = input_sol
        self.verbose = verbose
        self.kwargs = kwargs

    @staticmethod
    def unfold(data):
        """
        Method for unfolding second order tensors.

        **Input:**

        * **data** (`ndarray`) or (`list`):
            Input second order tensor to be unfolded.

        **Output/Returns:**

        * **M0, M1, M2** (`ndarrays`):
            Returns the 2-dimensional unfolded matrices.
        """

        if type(data) == list:
            x, y, z = data[0].shape[0], data[0].shape[1], len(data)
            data_ = np.zeros((x, y, z))
            for i in range(z):
                data_[:, :, i] = data[i]
            del data
            data = np.copy(data_)

        d0, d1, d2 = [0, 2, 1], [1, 2, 0], [2, 0, 1]
        z0, z1, z2 = np.transpose(data, d0), np.transpose(data, d1), np.transpose(data, d2)

        m0 = z0.reshape(data.shape[0], data.shape[2] * data.shape[1])
        m1 = z1.reshape(data.shape[1], data.shape[2] * data.shape[0])
        m2 = z2.reshape(data.shape[2], data.shape[0] * data.shape[1])

        return m0, m1, m2
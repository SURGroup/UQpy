import numpy as np
from UQpy.DimensionReduction.baseclass import POD

########################################################################################################################
########################################################################################################################
#                            Higher Order Singular Value Decomposition                                                 #
########################################################################################################################
########################################################################################################################


class HOSVD(POD):
    """
    HOSVD child class is used for higher-order singular value decomposition on the input solutions tensor.

    **Inputs:**

    * **input_sol** (`ndarray`) or (`list`):
        Second order tensor or list containing the solution snapshots. Third dimension or length of list corresponds
        to the number of snapshots.

    * **modes** (`int`):
        Number of modes to keep for dataset reconstruction.

    * **reconstr_perc** (`float`):
        Dataset reconstruction percentage

    **Methods:**
    """

    def __init__(self, input_sol, modes=10**10, reconstr_perc=10**10, verbose=False):

        super().__init__(input_sol, verbose)
        self.verbose = verbose
        self.modes = modes
        self.reconstr_perc = reconstr_perc

    def run(self, get_error=False):
        """
        Executes the HOSVD method in the ''HOSVD'' class.

        **Input:**

        * **get_error** (`Boolean`):
            A boolean declaring whether to return the reconstruction error.

        **Output/Returns:**

        * **reconstructed_solutions** (`ndarray`):
            Second order tensor containing the reconstructed solution snapshots in their initial spatial and
            temporal dimensions.

        * **reduced_solutions** (`ndarray`):
            An array containing the reduced solutions snapshots. The array's dimensions depends on the dimensions
            of input second order tensor and not on the input number of modes or reconstruction percentage.

        """

        if type(self.input_sol) == list:
            x, y, z = self.input_sol[0].shape[0], self.input_sol[0].shape[1], len(self.input_sol)
        else:
            x, y, z = self.input_sol.shape[0], self.input_sol.shape[1], self.input_sol.shape[2]

        a1, a2, a3 = POD.unfold(self.input_sol)

        u1, sig_1, v1 = np.linalg.svd(a1, full_matrices=True)
        u2, sig_2, v2 = np.linalg.svd(a2, full_matrices=True)
        u3, sig_3, v3 = np.linalg.svd(a3, full_matrices=True)

        sig_3_ = np.diag(sig_3)
        hold = np.dot(np.linalg.inv(u3), a3)
        kron_ = np.kron(u1, u2)

        s3 = np.array(np.dot(hold, np.linalg.inv(kron_.T)))

        if self.modes <= 0:
            print('Warning: Invalid input, the number of modes must be positive.')
            return [], []

        elif self.reconstr_perc <= 0:
            print('Warning: Invalid input, the reconstruction percentage is defined in the range (0,100].')
            return [], []

        elif self.modes != 10**10 and self.reconstr_perc != 10**10:
            print('Warning: Either a number of modes or a reconstruction percentage must be chosen, not both.')
            return [], []

        elif type(self.modes) != int:
            print('Warning: The number of modes must be an integer.')
            return [], []

        else:

            if self.modes == 10**10:
                error_ = []
                for i in range(0, x):
                    error_.append(np.sqrt(((sig_3[i + 1:]) ** 2).sum()) / np.sqrt((sig_3 ** 2).sum()))
                    if i == x:
                        error_.append(0)
                error = [i * 100 for i in error_]
                error.reverse()
                perc = error.copy()
                percentage = min(perc, key=lambda x: abs(x-self.reconstr_perc))
                self.modes = perc.index(percentage) + 1

            else:
                if self.modes > x:
                    print("Warning: A number of modes greater than the number of temporal dimensions was given.")
                    print("Number of temporal dimensions is {}.".format(x))

            reduced_solutions = np.dot(u3, sig_3_)
            u3hat = np.dot(u3[:, :self.modes], sig_3_[:self.modes, :self.modes])
            s3hat = np.dot(np.linalg.inv(sig_3_[:self.modes, :self.modes]), s3[:self.modes, :])

            b = np.kron(u1, u2)
            c = np.dot(s3hat, b.T)
            d = np.dot(u3hat[:, :], c)

            reconstructed_solutions = np.zeros((x, y, z))
            for i in range(z):
                reconstructed_solutions[0:x, 0:y, i] = d[i, :].reshape((x, y))

            if self.verbose:
                print("UQpy: Successful execution of HOSVD!")

            if self.modes == 10**10:
                if self.verbose:
                    print('Dataset reconstruction: {0:.3%}'.format(percentage / 100))

            else:
                if get_error:
                    error_rec = np.sqrt(((sig_3[self.modes:]) ** 2).sum()) / np.sqrt((sig_3 ** 2).sum())
                    print("Reduced-order reconstruction error: {0:.3%}".format(error_rec))

            return reconstructed_solutions, reduced_solutions
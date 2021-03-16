import numpy as np
from UQpy.DimensionReduction.baseclass import POD

########################################################################################################################
########################################################################################################################
#                                                     Snapshot POD                                                     #
########################################################################################################################
########################################################################################################################

class SnapshotPOD(POD):
    """
    Snapshot POD child class generates a set of temporal modes and spatial coefficients to approximate the solution.
    (Faster that direct POD)

    **Input:**

    * **input_sol** (`ndarray`) or (`list`):
        Second order tensor or list containing the solution snapshots. Third dimension or length of list corresponds
        to the number of snapshots.

    * **modes** (`int`):
        Number of POD modes used to approximate the input solution. Must be less than or equal
        to the number of grid points.

    * **reconstr_perc** (`float`):
        Dataset reconstruction percentage.

    **Methods:**
   """

    def __init__(self, input_sol, modes=10**10, reconstr_perc=10**10, verbose=False):

        super().__init__(input_sol, verbose)
        self.verbose = verbose
        self.modes = modes
        self.reconstr_perc = reconstr_perc

    def run(self):
        """
        Executes the Snapshot POD method in the ''Snapshot'' class.

        **Output/Returns:**

        * **reconstructed_solutions** (`ndarray`):
            Second order tensor containing the reconstructed solution snapshots in their initial spatial and
            temporal dimensions.

        * **reduced_solutions** (`ndarray`):
            An array containing the solution snapshots reduced in the temporal dimension.

        """
        if type(self.input_sol) == list:

            x, y, z = self.input_sol[0].shape[0], self.input_sol[0].shape[1], len(self.input_sol)
            u = np.zeros((z, x * y))

            for i in range(z):
                u[i, :] = self.input_sol[i].ravel()

        else:
            x, y, z = self.input_sol.shape[0], self.input_sol.shape[1], self.input_sol.shape[2]
            u = np.zeros((z, x * y))

            for i in range(z):
                u[i, :] = self.input_sol[:, :, i].ravel()

        c_s = np.dot(u, u.T) / (z - 1)

        eigval, a_s = np.linalg.eig(c_s)
        a_s = a_s.real
        eigval_ = eigval.real

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

            perc = []
            for i in range(z):
                perc.append((eigval_[:i + 1].sum() / eigval_.sum()) * 100)

            percentage = min(perc, key=lambda x: abs(x - self.reconstr_perc))

            if self.modes == 10**10:

                self.modes = perc.index(percentage) + 1

            else:

                if self.modes > z:
                    print("Warning: A number of modes greater than the number of dimensions was given.")
                    print("Number of dimensions is {}".format(z))

            phi_s = np.dot(u.T, a_s)
            reconstructed_solutions_ = np.dot(a_s[:, :self.modes], phi_s[:, :self.modes].T)
            reduced_solutions_ = (np.dot(u.T, a_s[:, :self.modes])).T

            reconstructed_solutions = np.zeros((x, y, z))
            reduced_solutions = np.zeros((x, y, self.modes))

            for i in range(z):
                reconstructed_solutions[0:x, 0:y, i] = reconstructed_solutions_[i, :].reshape((x, y))

            for i in range(self.modes):
                reduced_solutions[0:x, 0:y, i] = reduced_solutions_[i, :].reshape((x, y))

            if self.verbose:
                print("UQpy: Successful execution of Snapshot POD!")

            if self.verbose:
                print('Dataset reconstruction: {:.3%}'.format(perc[self.modes - 1] / 100))

            return reconstructed_solutions, reduced_solutions
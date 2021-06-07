import numpy as np
from UQpy.dimension_reduction.baseclass import POD


class DirectPOD(POD):
    """
    Direct POD child class generates a set of spatial modes and time coefficients to approximate the solution.

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

    def __init__(self, solution_snapshots, modes=10 ** 10, reconstruction_percentage=10 ** 10, verbose=False):

        super().__init__(solution_snapshots, verbose)
        self.verbose = verbose
        self.modes = modes
        self.reconstruction_percentage = reconstruction_percentage

    def run(self):

        """
        Executes the Direct POD method in the ''Direct'' class.

        **Output/Returns:**

        * **reconstructed_solutions** (`ndarray`):
            Second order tensor containing the reconstructed solution snapshots in their initial spatial and
            temporal dimensions.

        * **reduced_solutions** (`ndarray`):
            An array containing the solution snapshots reduced in the spatial dimension.

        """

        if type(self.solution_snapshots) == list:
            rows = self.solution_snapshots[0].shape[0]
            columns = self.solution_snapshots[0].shape[1]
            snapshot_number = len(self.solution_snapshots)
            u = np.zeros((snapshot_number, rows * columns))

            for i in range(snapshot_number):
                u[i, :] = self.solution_snapshots[i].ravel()

        else:
            rows = self.solution_snapshots.shape[0]
            columns = self.solution_snapshots.shape[1]
            snapshot_number = self.solution_snapshots.shape[2]
            u = np.zeros((snapshot_number, rows * columns))

            for i in range(snapshot_number):
                u[i, :] = self.solution_snapshots[:, :, i].ravel()

        c = np.dot(u.T, u) / (snapshot_number - 1)
        eigenvalues, phi = np.linalg.eig(c)
        phi = phi.real
        real_eigenvalues = eigenvalues.real
        a = np.dot(u, phi)

        if self.modes <= 0:
            print('Warning: Invalid input, the number of modes must be positive.')
            return [], []

        elif self.reconstruction_percentage <= 0:
            print('Warning: Invalid input, the reconstruction percentage is defined in the range (0,100].')
            return [], []

        elif self.modes != 10**10 and self.reconstruction_percentage != 10**10:
            print('Warning: Either a number of modes or a reconstruction percentage must be chosen, not both.')
            return [], []

        elif type(self.modes) != int:
            print('Warning: The number of modes must be an integer.')
            return [], []

        else:

            percentages = []
            for i in range(rows * columns):
                percentages.append((real_eigenvalues[:i + 1].sum() / real_eigenvalues .sum()) * 100)

            minimum_percentage = min(percentages, key=lambda x: abs(x - self.reconstruction_percentage))

            if self.modes == 10**10:
                self.modes = percentages.index(minimum_percentage) + 1
            else:
                if self.modes > rows * columns:
                    print("Warning: A number of modes greater than the number of dimensions was given.")
                    print("Number of dimensions is {}".format(rows * columns))

            reconstructed_solutions_ = np.dot(a[:, :self.modes], phi[:, :self.modes].T)
            reduced_solutions = np.dot(u, phi[:, :self.modes])

            reconstructed_solutions = np.zeros((rows, columns, snapshot_number ))
            for i in range(snapshot_number):
                reconstructed_solutions[0:rows, 0:columns, i] = reconstructed_solutions_[i, :].reshape((rows , columns))

            if self.verbose:
                print("UQpy: Successful execution of Direct POD!")
                if snapshot_number < rows * columns and rows * columns > 1000:
                    print("Snapshot POD is recommended.")

            if self.verbose:
                print('Dataset reconstruction: {:.3%}'.format(percentages[self.modes - 1] / 100))

            return reconstructed_solutions, reduced_solutions

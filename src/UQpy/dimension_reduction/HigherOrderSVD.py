import logging

import numpy as np
from UQpy.dimension_reduction.baseclass import POD


class HigherOrderSVD(POD):
    """
    HigherOrderSVD child class is used for higher-order singular value decomposition on the input solutions tensor.

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

    def __init__(self, solution_snapshots, modes=10 ** 10, reconstruction_percentage=10 ** 10):

        super().__init__(solution_snapshots)
        self.logger = logging.getLogger(__name__)
        self.modes = modes
        self.reconstruction_percentage = reconstruction_percentage

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

        if type(self.solution_snapshots) == list:
            rows = self.solution_snapshots[0].shape[0]
            columns = self.solution_snapshots[0].shape[1]
            snapshot_number = len(self.solution_snapshots)
        else:
            rows = self.solution_snapshots.shape[0]
            columns = self.solution_snapshots.shape[1]
            snapshot_number = self.solution_snapshots.shape[2]

        a1, a2, a3 = POD.unfold(self.solution_snapshots)

        u1, sig_1, v1 = np.linalg.svd(a1, full_matrices=True)
        u2, sig_2, v2 = np.linalg.svd(a2, full_matrices=True)
        u3, sig_3, v3 = np.linalg.svd(a3, full_matrices=True)

        sig_3_ = np.diag(sig_3)
        hold = np.dot(np.linalg.inv(u3), a3)
        kronecker_product = np.kron(u1, u2)

        s3 = np.array(np.dot(hold, np.linalg.inv(kronecker_product.T)))

        if self.modes <= 0:
            self.logger.warning('Invalid input, the number of modes must be positive.')
            return [], []

        elif self.reconstruction_percentage <= 0:
            self.logger.warning('Invalid input, the reconstruction percentage is defined in the range (0,100].')
            return [], []

        elif self.modes != 10**10 and self.reconstruction_percentage != 10**10:
            self.logger.warning('Either a number of modes or a reconstruction percentage must be chosen, not both.')
            return [], []

        elif type(self.modes) != int:
            self.logger.warning('The number of modes must be an integer.')
            return [], []

        else:

            if self.modes == 10**10:
                error_ = []
                for i in range(0, rows):
                    error_.append(np.sqrt(((sig_3[i + 1:]) ** 2).sum()) / np.sqrt((sig_3 ** 2).sum()))
                    if i == rows:
                        error_.append(0)
                error = [i * 100 for i in error_]
                error.reverse()
                perc = error.copy()
                percentage = min(perc, key=lambda x: abs(x - self.reconstruction_percentage))
                self.modes = perc.index(percentage) + 1

            else:
                if self.modes > rows:
                    self.logger.warning("A number of modes greater than the number of temporal dimensions was given."
                                        "Number of temporal dimensions is {}.".format(rows))

            reduced_solutions = np.dot(u3, sig_3_)
            u3hat = np.dot(u3[:, :self.modes], sig_3_[:self.modes, :self.modes])
            s3hat = np.dot(np.linalg.inv(sig_3_[:self.modes, :self.modes]), s3[:self.modes, :])

            b = np.kron(u1, u2)
            c = np.dot(s3hat, b.T)
            d = np.dot(u3hat[:, :], c)

            reconstructed_solutions = np.zeros((rows, columns, snapshot_number))
            for i in range(snapshot_number):
                reconstructed_solutions[0:rows, 0:columns, i] = d[i, :].reshape((rows, columns))

            self.logger.info("UQpy: Successful execution of HOSVD!")

            if self.modes == 10**10:
                self.logger.info('Dataset reconstruction: {0:.3%}'.format(percentage / 100))

            else:
                if get_error:
                    error_rec = np.sqrt(((sig_3[self.modes:]) ** 2).sum()) / np.sqrt((sig_3 ** 2).sum())
                    self.logger.warning("Reduced-order reconstruction error: {0:.3%}".format(error_rec))

            return reconstructed_solutions, reduced_solutions

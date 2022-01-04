import logging
from typing import Union

import numpy as np


class SnapshotPOD:

    def __init__(
        self,
        solution_snapshots: Union[np.ndarray, list],
        modes: int = 10 ** 10,
        reconstruction_percentage: float = 10 ** 10,
    ):
        """
        Snapshot POD child class generates a set of temporal modes and spatial coefficients to approximate the solution.
        (Faster that direct POD)

        :param solution_snapshots: Second order tensor or list containing the solution snapshots. Third dimension or
         length of list corresponds to the number of snapshots.
        :param modes: Number of POD modes used to approximate the input solution. Must be less than or equal
         to the number of grid points.
        :param reconstruction_percentage: Dataset reconstruction percentage.
        """
        self.solution_snapshots = solution_snapshots
        self.logger = logging.getLogger(__name__)
        self.modes = modes
        self.reconstruction_percentage = reconstruction_percentage

    def run(self):
        """
        Executes the Snapshot POD method.

        :return: Second order tensor containing the reconstructed solution snapshots in their initial spatial and
         temporal dimensions and an array containing the solution snapshots reduced in the temporal dimension.
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

        c_s = np.dot(u, u.T) / (snapshot_number - 1)

        eigenvalues, a_s = np.linalg.eig(c_s)
        a_s = a_s.real
        real_eigenvalues = eigenvalues.real

        if self.modes <= 0:
            self.logger.warning("Invalid input, the number of modes must be positive.")
            return [], []

        elif self.reconstruction_percentage <= 0:
            self.logger.warning(
                "Invalid input, the reconstruction percentage is defined in the range (0,100]."
            )
            return [], []

        elif self.modes != 10 ** 10 and self.reconstruction_percentage != 10 ** 10:
            self.logger.warning(
                "Either a number of modes or a reconstruction percentage must be chosen, not both."
            )
            return [], []

        elif type(self.modes) != int:
            self.logger.warning("The number of modes must be an integer.")
            return [], []

        else:

            percentages = []
            for i in range(snapshot_number):
                percentages.append(
                    (real_eigenvalues[: i + 1].sum() / real_eigenvalues.sum()) * 100
                )

            minimum_percentage = min(
                percentages, key=lambda x: abs(x - self.reconstruction_percentage)
            )

            if self.modes == 10 ** 10:
                self.modes = percentages.index(minimum_percentage) + 1
            else:
                if self.modes > snapshot_number:
                    self.logger.warning(
                        "A number of modes greater than the number of dimensions was given."
                        "Number of dimensions is {}".format(snapshot_number)
                    )

            phi_s = np.dot(u.T, a_s)
            reconstructed_solutions_ = np.dot(
                a_s[:, : self.modes], phi_s[:, : self.modes].T
            )
            reduced_solutions_ = (np.dot(u.T, a_s[:, : self.modes])).T

            reconstructed_solutions = np.zeros((rows, columns, snapshot_number))
            reduced_solutions = np.zeros((rows, columns, self.modes))

            for i in range(snapshot_number):
                reconstructed_solutions[
                    0:rows, 0:columns, i
                ] = reconstructed_solutions_[i, :].reshape((rows, columns))

            for i in range(self.modes):
                reduced_solutions[0:rows, 0:columns, i] = reduced_solutions_[
                    i, :
                ].reshape((rows, columns))

            self.logger.info("UQpy: Successful execution of Snapshot POD!")

            self.logger.info(
                "Dataset reconstruction: {:.3%}".format(
                    percentages[self.modes - 1] / 100
                )
            )

            return reconstructed_solutions, reduced_solutions

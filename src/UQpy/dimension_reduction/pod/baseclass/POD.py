import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from beartype import beartype

from UQpy.utilities.ValidationTypes import PositiveInteger, PositiveFloat


class POD(ABC):
    @beartype
    def __init__(self,
                 solution_snapshots: Union[np.ndarray, list],
                 modes: PositiveInteger = 10 ** 10,
                 reconstruction_percentage: Union[PositiveInteger, PositiveFloat] = 10 ** 10):
        """

        :param solution_snapshots: Second order tensor or list containing the solution snapshots. Third dimension or
         length of list corresponds to the number of snapshots.
        :param modes: Number of POD modes used to approximate the input solution. Must be less than or equal to the
         number of grid points.
        :param reconstruction_percentage: Dataset reconstruction percentage.
        """
        self.reduced_solution = None
        """Second order tensor containing the reconstructed solution snapshots in their initial spatial and temporal 
        dimensions."""
        self.reconstructed_solution = None
        """An array containing the solution snapshots reduced in the spatial dimension."""
        self.logger = logging.getLogger(__name__)
        if reconstruction_percentage <= 0:
            raise ValueError("Invalid input, the reconstruction percentage is defined in the range (0,100].")

        if modes != 10 ** 10 and reconstruction_percentage != 10 ** 10:
            raise ValueError("Either a number of modes or a reconstruction percentage must be chosen, not both.")

        self.solution_snapshots = solution_snapshots
        self.logger = logging.getLogger(__name__)
        self.modes = modes
        self.reconstruction_percentage = reconstruction_percentage

    def check_input(self):
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
        return columns, rows, snapshot_number, u

    @abstractmethod
    def _calculate_c_and_iterations(self, u, snapshot_number, rows, columns):
        pass

    @abstractmethod
    def _calculate_reduced_and_reconstructed_solutions(self, u, phi, rows, columns, snapshot_number):
        pass

    def run(self):
        """
        Executes the POD method. Since :class:`.POD` is an abstract baseclass, the one of the concrete implementations
        will be executed

        :return: Second order tensor containing the reconstructed solution snapshots in their initial spatial and
         temporal dimensions and an array containing the solution snapshots reduced in the spatial dimension.
        """

        columns, rows, snapshot_number, u = self.check_input()

        c, n_iterations = self._calculate_c_and_iterations(u, snapshot_number, rows, columns)

        eigenvalues, phi = np.linalg.eig(c)
        phi = phi.real
        real_eigenvalues = eigenvalues.real

        percentages = [(real_eigenvalues[: i + 1].sum() / real_eigenvalues.sum()) * 100 for i in range(n_iterations)]

        minimum_percentage = min(percentages, key=lambda x: abs(x - self.reconstruction_percentage))

        if self.modes == 10 ** 10:
            self.modes = percentages.index(minimum_percentage) + 1
        elif self.modes > n_iterations:
            self.logger.warning(
                "A number of modes greater than the number of dimensions was given."
                "Number of dimensions is %i", n_iterations)

        reconstructed_solutions, reduced_solutions = \
            self._calculate_reduced_and_reconstructed_solutions(u, phi, rows, columns, snapshot_number)

        self.logger.info(f"UQpy: Successful execution of {type(self).__name__}!")

        self.logger.info("Dataset reconstruction: {:.3%}".format(percentages[self.modes - 1] / 100))

        self.reconstructed_solution = reconstructed_solutions
        self.reduced_solution = reduced_solutions

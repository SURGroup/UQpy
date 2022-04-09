import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from beartype import beartype

from UQpy.utilities.ValidationTypes import PositiveInteger, PositiveFloat


class POD(ABC):
    @beartype
    def __init__(self,
                 solution_snapshots: Union[np.ndarray, list] = None,
                 n_modes: PositiveInteger = None,
                 reconstruction_percentage: Union[PositiveInteger, PositiveFloat] = None):
        """

        :param solution_snapshots: Array or list containing the solution snapshots. If provided as an
         :class:`numpy.ndarray`, it should be three-dimensional, where the third dimension of the array corresponds to
         the number of snapshots. If provided as a list, the length of the list corresponds to the number of snapshots.

         If `solution_snapshots` is provided, the :py:meth:`.run` method will be executed automatically. If it is not
         provided, then the :py:meth:`.run` method must be executed manually and provided with `solution_snapshots`.
        :param n_modes: Number of POD modes used to approximate the input solution. Must be less than or equal to the
         number of dimensions in a snapshot. Either `n_modes` or `reconstruction_percentage` must be provided, but not
         both.
        :param reconstruction_percentage: Specified dataset reconstruction percentage. Must be between 0 and 100. Either
            `n_modes` or `reconstruction_percentage` must be provided, but not both.
        """
        self.U = None
        """Two dimensional dataset constructed by :class:`.POD`"""
        self.eigenvalues = None
        """Eigenvalues produced by the POD decomposition of the solution snapshots."""
        self.phi = None
        """Eigenvectors produced by the POD decomposition of the solution snapshots."""
        self.reduced_solution = None
        """Second order tensor containing the reconstructed solution snapshots in their initial spatial and temporal 
        dimensions."""
        self.reconstructed_solution = None
        """An array containing the solution snapshots reduced in the spatial dimension."""
        self.logger = logging.getLogger(__name__)

        if n_modes is not None and reconstruction_percentage is not None:
            raise ValueError("Either a number of modes or a reconstruction percentage must be chosen, not both.")

        if reconstruction_percentage is not None and reconstruction_percentage <= 0:
            raise ValueError("Invalid input, the reconstruction percentage is defined in the range (0,100].")



        self.solution_snapshots = solution_snapshots
        self.logger = logging.getLogger(__name__)
        self.modes = n_modes
        self.reconstruction_percentage = reconstruction_percentage

        if reconstruction_percentage is None:
            self.reconstruction_percentage = 10**10

        if solution_snapshots is not None:
            self.run(solution_snapshots)

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

    def run(self, solution_snapshots: Union[np.ndarray, list]):
        """
        Executes the POD method. Since :class:`.POD` is an abstract baseclass, one of the concrete implementations
        will be executed.

        :param solution_snapshots: Array or list containing the solution snapshots. If provided as an
         :class:`numpy.ndarray`, it should be three-dimensional, where the third dimension of the array corresponds to
         the number of snapshots. If provided as a list, the length of the list corresponds to the number of snapshots.

         If `solution_snapshots` is provided, the :py:meth:`.run` method will be executed automatically. If it is not
         provided, then the :py:meth:`.run` method must be executed manually and provided with `solution_snapshots`.
        """

        columns, rows, snapshot_number, self.U = self.check_input()

        c, n_iterations = self._calculate_c_and_iterations(self.U, snapshot_number, rows, columns)

        complex_eigenvalues, phi = np.linalg.eig(c)
        self.phi = phi.real

        self.eigenvalues = complex_eigenvalues.real

        percentages = [(self.eigenvalues[: i + 1].sum() / self.eigenvalues.sum()) * 100 for i in range(n_iterations)]

        minimum_percentage = min(percentages, key=lambda x: abs(x - self.reconstruction_percentage))

        if self.modes is None:
            self.modes = percentages.index(minimum_percentage) + 1
        elif self.modes > n_iterations:
            self.logger.warning(
                "A number of modes greater than the number of dimensions was given."
                "Number of dimensions is %i", n_iterations)

        reconstructed_solutions, reduced_solutions = \
            self._calculate_reduced_and_reconstructed_solutions(self.U, phi, rows, columns, snapshot_number)

        self.logger.info(f"UQpy: Successful execution of {type(self).__name__}!")

        self.logger.info("Dataset reconstruction: {:.3%}".format(percentages[self.modes - 1] / 100))

        self.reconstructed_solution = reconstructed_solutions
        self.reduced_solution = reduced_solutions

import numpy as np
from UQpy.dimension_reduction.pod.baseclass.POD import POD


class SnapshotPOD(POD):

    def run(self, solution_snapshots):
        """
        Executes proper orthogonal decomposition using the :class:`.SnapshotPOD` algorithm.
        """
        return super().run(solution_snapshots)

    def _calculate_reduced_and_reconstructed_solutions(self, u, phi, rows, columns, snapshot_number):
        phi_s = np.dot(u.T, phi)
        reconstructed_solutions_ = np.dot(phi[:, : self.modes], phi_s[:, : self.modes].T)
        reduced_solutions_ = (np.dot(u.T, phi[:, : self.modes])).T

        reconstructed_solutions = np.zeros((rows, columns, snapshot_number))
        reduced_solutions = np.zeros((rows, columns, self.modes))

        for i in range(snapshot_number):
            reconstructed_solutions[0:rows, 0:columns, i] = reconstructed_solutions_[i, :].reshape((rows, columns))

        for i in range(self.modes):
            reduced_solutions[0:rows, 0:columns, i] = reduced_solutions_[i, :].reshape((rows, columns))

        return reconstructed_solutions, reduced_solutions

    def _calculate_c_and_iterations(self, u, snapshot_number, rows, columns):
        c = np.dot(u, u.T) / (snapshot_number - 1)
        n_iterations = snapshot_number
        return c, n_iterations

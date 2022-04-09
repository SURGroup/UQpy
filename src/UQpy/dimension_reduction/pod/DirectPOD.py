import numpy as np
from UQpy.dimension_reduction.pod.baseclass.POD import POD


class DirectPOD(POD):

    def run(self, solution_snapshots):
        """
        Executes proper orthogonal decomposition using the :class:`.DirectPOD` algorithm.
        """
        return super().run(solution_snapshots)

    def _calculate_c_and_iterations(self, u, snapshot_number, rows, columns):
        if snapshot_number < rows * columns and rows * columns > 1000:
            self.logger.warning("Snapshot POD is recommended.")

        c = np.dot(u.T, u) / (snapshot_number - 1)
        n_iterations = rows * columns
        return c, n_iterations

    def _calculate_reduced_and_reconstructed_solutions(self, u, phi, rows, columns, snapshot_number):
        a = np.dot(u, phi)
        reconstructed_solutions_ = np.dot(a[:, : self.modes], phi[:, : self.modes].T)
        reduced_solutions = np.dot(u, phi[:, : self.modes])
        reconstructed_solutions = np.zeros((rows, columns, snapshot_number))
        for i in range(snapshot_number):
            reconstructed_solutions[0:rows, 0:columns, i] = reconstructed_solutions_[i, :].reshape((rows, columns))
        return reconstructed_solutions, reduced_solutions

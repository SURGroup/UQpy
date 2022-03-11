import logging

from beartype import beartype

from UQpy.utilities.ValidationTypes import *
import numpy as np


class HigherOrderSVD:
    @beartype
    def __init__(
            self,
            solution_snapshots: Union[np.ndarray, list],
            modes: PositiveInteger = 10 ** 10,
            reconstruction_percentage: Union[PositiveFloat, PositiveInteger] = 10 ** 10,
    ):
        """

        :param solution_snapshots: Second order tensor or list containing the solution snapshots. Third dimension or
         length of list corresponds to the number of snapshots.
        :param modes: Number of modes to keep for dataset reconstruction.
        :param reconstruction_percentage: Dataset reconstruction percentage.
        """
        self.reconstruction_error = None
        self.s3hat: Numpy2DFloatArray = None
        """Normalized core tensor produced by the HOSVD decomposition."""
        self.u3hat: Numpy2DFloatArray = None
        """Normalized unitary array produced by the HOSVD decomposition"""
        self.reduced_solutions = None
        self.s3 = None
        self.v3 = None
        self.sig_3 = None
        self.u3 = None
        self.v2 = None
        self.sig_2 = None
        self.u2: Numpy2DFloatArray = None
        """Unitary array of the SVD of the second unfolded matrix."""
        self.v1 = None
        self.sig_1 = None
        self.u1: Numpy2DFloatArray = None
        """Unitary array of the SVD of the first unfolded matrix"""
        self.solution_snapshots = solution_snapshots
        self.logger = logging.getLogger(__name__)
        self.modes = modes
        self.reconstruction_percentage = reconstruction_percentage

        if self.modes != 10 ** 10 and self.reconstruction_percentage != 10 ** 10:
            raise ValueError("Either a number of modes or a reconstruction percentage must be chosen, not both.")

        if self.solution_snapshots is not None:
            self.factorize(get_error=True)

    def factorize(self, get_error: bool = False):
        """
        Executes the HOSVD method.

        :param get_error: A boolean declaring whether to return the reconstruction error.
        """
        rows = self.solution_snapshots[0].shape[0]

        a1, a2, a3 = HigherOrderSVD.unfold3d(self.solution_snapshots)

        self.u1, self.sig_1, self.v1 = np.linalg.svd(a1, full_matrices=True)
        self.u2, self.sig_2, self.v2 = np.linalg.svd(a2, full_matrices=True)
        self.u3, self.sig_3, self.v3 = np.linalg.svd(a3, full_matrices=True)

        sig_3_ = np.diag(self.sig_3)
        hold = np.dot(np.linalg.inv(self.u3), a3)
        kronecker_product = np.kron(self.u1, self.u2)
        self.s3 = np.array(np.dot(hold, np.linalg.inv(kronecker_product.T)))

        if self.modes == 10 ** 10:
            error_ = []
            for i in range(rows):
                error_.append(np.sqrt(((self.sig_3[i + 1:]) ** 2).sum()) / np.sqrt((self.sig_3 ** 2).sum()))
                if i == rows:
                    error_.append(0)
            error = [i * 100 for i in error_]
            error.reverse()
            perc = error.copy()
            percentage = min(perc, key=lambda x: abs(x - self.reconstruction_percentage))
            self.modes = perc.index(percentage) + 1

        elif self.modes > rows:
            self.logger.warning(
                "A number of modes greater than the number of temporal dimensions was given."
                "Number of temporal dimensions is {}.".format(rows))

        self.reduced_solutions = np.dot(self.u3, sig_3_)
        self.u3hat = np.dot(self.u3[:, : self.modes], sig_3_[: self.modes, : self.modes])
        self.s3hat = np.dot(np.linalg.inv(sig_3_[: self.modes, : self.modes]), self.s3[: self.modes, :])

        if self.modes == 10 ** 10:
            self.logger.info("Dataset reconstruction: {0:.3%}".format(self.reconstruction_percentage / 100))

        elif get_error:
            self.reconstruction_error = np.sqrt(((self.s3hat[self.modes:]) ** 2).sum()) / np.sqrt(
                (self.sig_3 ** 2).sum())
            self.logger.warning("Reduced-order reconstruction error: {0:.3%}".format(self.reconstruction_error))

    @staticmethod
    def unfold3d(second_order_tensor: np.ndarray):
        """

        :param second_order_tensor: A three-dimensional :class:`ndarray` which contains the data to be unfolded.
        :returns: Three unfolded matrices in the form of two-dimensional :class:`ndarray`.
        """
        if type(second_order_tensor) == list:
            rows = second_order_tensor[0].shape[0]
            columns = second_order_tensor[0].shape[1]
            number_of_data = len(second_order_tensor)
            tensor_of_list = np.zeros((rows, columns, number_of_data))
            for i in range(number_of_data):
                tensor_of_list[:, :, i] = second_order_tensor[i]
            del second_order_tensor
            second_order_tensor = np.copy(tensor_of_list)

        permutation1 = [0, 2, 1]
        permutation2 = [1, 2, 0]
        permutation3 = [2, 0, 1]

        permuted_tensor1 = np.transpose(second_order_tensor, permutation1)
        permuted_tensor2 = np.transpose(second_order_tensor, permutation2)
        permuted_tensor3 = np.transpose(second_order_tensor, permutation3)

        matrix1 = permuted_tensor1.reshape(second_order_tensor.shape[0],
                                           second_order_tensor.shape[2] * second_order_tensor.shape[1], )
        matrix2 = permuted_tensor2.reshape(second_order_tensor.shape[1],
                                           second_order_tensor.shape[2] * second_order_tensor.shape[0], )
        matrix3 = permuted_tensor3.reshape(second_order_tensor.shape[2],
                                           second_order_tensor.shape[0] * second_order_tensor.shape[1], )

        return matrix1, matrix2, matrix3

    @staticmethod
    def reconstruct(u1: Numpy2DFloatArray,
                    u2: Numpy2DFloatArray,
                    u3hat: Numpy2DFloatArray,
                    s3hat: Numpy2DFloatArray):
        """
        Reconstructs the approximated solution.

        :param u1: Unitary array of the SVD of the first unfolded matrix.
        :param u2: Unitary array of the SVD of the second unfolded matrix.
        :param u3hat: Normalized unitary array produced by the HOSVD decomposition.
        :param s3hat: Normalized core tensor produced by the HOSVD decomposition
        :return: An :class:`ndarray` containing the reconstruction of the approximated solution.
        """
        b = np.kron(u1, u2)
        c = np.dot(s3hat, b.T)
        d = np.dot(u3hat[:, :], c)

        rows = u1.shape[0]
        columns = u2.shape[0]
        snapshot_number = u3hat.shape[0]

        reconstructed_solutions = np.zeros((rows, columns, snapshot_number))
        for i in range(snapshot_number):
            reconstructed_solutions[0:rows, 0:columns, i] = d[i, :].reshape((rows, columns))

        return reconstructed_solutions

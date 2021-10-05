import logging
from UQpy.dimension_reduction.SVD import SVD
from UQpy.utilities.ValidationTypes import *
import numpy as np


class HigherOrderSVD:

    def __init__(self,
                 solution_snapshots: Union[np.ndarray, list],
                 modes: PositiveInteger = 10 ** 10,
                 reconstruction_percentage: float = 10 ** 10):

        self.solution_snapshots = solution_snapshots
        self.logger = logging.getLogger(__name__)
        self.modes = modes
        self.reconstruction_percentage = reconstruction_percentage

    def factorize(self, get_error: bool = False):

        if self.modes <= 0:
            print('Warning: Invalid input, the number of modes must be positive.')
            return [], [], [], [], [], []

        elif self.reconstruction_percentage <= 0:
            print('Warning: Invalid input, the reconstruction percentage is defined in the range (0,100].')
            return [], [], [], [], [], []

        elif self.modes != 10**10 and self.reconstruction_percentage != 10**10:
            print('Warning: Either a number of modes or a reconstruction percentage must be chosen, not both.')
            return [], [], [], [], [], []

        elif type(self.modes) != int:
            print('Warning: The number of modes must be an integer.')
            return [], [], [], [], [], []

        rows = self.solution_snapshots[0].shape[0]

        a1, a2, a3 = HigherOrderSVD.unfold3d(self.solution_snapshots)

        u1, sig_1, v1 = np.linalg.svd(a1, full_matrices=True)
        u2, sig_2, v2 = np.linalg.svd(a2, full_matrices=True)
        u3, sig_3, v3 = np.linalg.svd(a3, full_matrices=True)

        sig_3_ = np.diag(sig_3)
        hold = np.dot(np.linalg.inv(u3), a3)
        kronecker_product = np.kron(u1, u2)
        s3 = np.array(np.dot(hold, np.linalg.inv(kronecker_product.T)))

        if self.modes != 10**10 and self.reconstruction_percentage != 10**10:
            self.logger.warning('Either a number of modes or a reconstruction percentage must be chosen, not both.')
            return [], [], [], [], [], []

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

            if self.modes == 10 ** 10:
                self.logger.info(
                    'Dataset reconstruction: {0:.3%}'.format(self.reconstruction_percentage / 100))

            else:
                if get_error:
                    error_rec = np.sqrt(((s3hat[self.modes:]) ** 2).sum()) / np.sqrt((sig_3 ** 2).sum())
                    self.logger.warning("Reduced-order reconstruction error: {0:.3%}".format(error_rec))

            return u1, u2, u3, s3, u3hat, s3hat, reduced_solutions

    @staticmethod
    def unfold3d(second_order_tensor):
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
                                           second_order_tensor.shape[2] * second_order_tensor.shape[1])
        matrix2 = permuted_tensor2.reshape(second_order_tensor.shape[1],
                                           second_order_tensor.shape[2] * second_order_tensor.shape[0])
        matrix3 = permuted_tensor3.reshape(second_order_tensor.shape[2],
                                           second_order_tensor.shape[0] * second_order_tensor.shape[1])

        return matrix1, matrix2, matrix3

    @staticmethod
    def reconstruct(u1, u2,  u3hat, s3hat):
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

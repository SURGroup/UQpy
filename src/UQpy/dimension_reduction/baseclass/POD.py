import logging
from typing import Union

import numpy as np


class POD:
    """
    Performs Direct and Snapshot Proper Orthogonal Decomposition (POD) as well as Higher-order Singular Value
    Decomposition (HOSVD) for dimension reduction of datasets.

    **Input:**

    * **input_sol** (`ndarray`) or (`list`):
        Second order tensor or list containing the solution snapshots. Third dimension or length of list corresponds
        to the number of snapshots.

    * **verbose** (`Boolean`):
        A boolean declaring whether to write text to the terminal.

    **Methods:**
    """

    def __init__(self,
                 solution_snapshots: Union[np.ndarray, list],
                 **kwargs):

        self.solution_snapshots = solution_snapshots
        self.kwargs = kwargs

    @staticmethod
    def unfold(second_order_tensor):
        """
        Method for unfolding second order tensors.

        **Input:**

        * **data** (`ndarray`) or (`list`):
            Input second order tensor to be unfolded.

        **Output/Returns:**

        * **M0, M1, M2** (`ndarrays`):
            Returns the 2-dimensional unfolded matrices.
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
                                           second_order_tensor.shape[2] * second_order_tensor.shape[1])
        matrix2 = permuted_tensor2.reshape(second_order_tensor.shape[1],
                                           second_order_tensor.shape[2] * second_order_tensor.shape[0])
        matrix3 = permuted_tensor3.reshape(second_order_tensor.shape[2],
                                           second_order_tensor.shape[0] * second_order_tensor.shape[1])

        return matrix1, matrix2, matrix3

from typing import Union
import numpy as np


class SVD:
    """
    TODO: Description of SVD
    """
    def __init__(self,
                 matrix: Union[np.ndarray, list],
                 tolerance: float = None,
                 rank: int = None,
                 full_matrices: bool = True,
                 hermitian: bool = False):

        self.matrix = matrix
        self.rank = rank
        self.tolerance = tolerance
        self.full_matrices = full_matrices
        self.hermitian = hermitian

    def run(self):
        ui, si, vi = np.linalg.svd(self.matrix, full_matrices=self.full_matrices, hermitian=self.hermitian)
        si = np.diag(si)
        vi = vi.T
        if self.rank is None:
            self.rank = np.linalg.matrix_rank(self.matrix) if self.tolerance is \
                                                              None else np.linalg.matrix_rank(self.matrix,
                                                                                              tol=self.tolerance)

        left_eigenvectors = ui[:, :self.rank]
        eigenvalues = si[:self.rank, :self.rank]
        right_eigenvectors = vi[:, :self.rank]

        return left_eigenvectors, eigenvalues, right_eigenvectors

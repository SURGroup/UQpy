import numpy as np


class SVD:
    """
    TODO: Description of SVD
    TODO: Add test
    """
    @staticmethod
    def factorize(matrix, rank=False, tolerance=None, full_matrices=True, hermitian=False):
        ui, si, vi = np.linalg.svd(matrix, full_matrices=full_matrices, hermitian=hermitian)
        si = np.diag(si)
        vi = vi.T
        if rank:
            rank = np.linalg.matrix_rank(matrix) if tolerance is None \
                else np.linalg.matrix_rank(matrix, tol=tolerance)
            left_eigenvectors = ui[:, :rank]
            eigenvalues = si[:rank, :rank]
            right_eigenvectors = vi[:, :rank]

        elif not rank:
            left_eigenvectors = ui
            eigenvalues = si
            right_eigenvectors = vi

        else:
            left_eigenvectors = ui[:, :rank]
            eigenvalues = si[:rank, :rank]
            right_eigenvectors = vi[:, :rank]

        return left_eigenvectors, eigenvalues, right_eigenvectors

    @staticmethod
    def reconstruct(u, s, v):
        return np.dot(np.dot(u, s), v.T)
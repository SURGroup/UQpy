import numpy as np


class QR:
    """
    TODO: Description of QR factorization
    TODO: Add test
    """
    @staticmethod
    def factorize(matrix, rank=None, mode='reduced'):
        # matrix size: MxN
        # if mode is "reduced" : q MxK, r KxN
        # if mode is "complete": q MxM, r MxN

        q, r = np.linalg.qr(matrix, mode=mode)

        if rank and mode is 'complete':
            orthonormal_matrix = q[:, :rank]
            upper_triangular = r[:rank, :rank]
        else:
            orthonormal_matrix = q
            upper_triangular = r

        return orthonormal_matrix, upper_triangular

    @staticmethod
    def reconstruct(q, r):
        return np.dot(q, r)

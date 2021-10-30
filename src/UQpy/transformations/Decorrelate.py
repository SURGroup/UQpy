import numpy as np
from beartype import beartype
from scipy.linalg import cholesky


class Decorrelate:
    @beartype
    def __init__(self, samples_z: np.ndarray, corr_z: np.ndarray):
        """
        A class to remove correlation from correlated standard normal random variables.

        :param samples_z: Correlated standard normal vector of shape ``(nsamples, dimension)``.
        :param corr_z: The correlation  matrix (:math:`\mathbf{C_Z}`) of the standard normal random vector **Z** .
        """
        self.samples_z = samples_z
        self.corr_z = corr_z
        self.H = cholesky(self.corr_z, lower=True)
        self.samples_u = np.linalg.solve(self.H, samples_z.T.squeeze()).T

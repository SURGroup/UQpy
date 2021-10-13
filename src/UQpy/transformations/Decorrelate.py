import numpy as np
from beartype import beartype
from scipy.linalg import cholesky


class Decorrelate:
    """
    A class to remove correlation from correlated standard normal random variables.


    **Inputs:**

    * **samples_z** (`ndarray`):
            Correlated standard normal vector of shape ``(nsamples, dimension)``.

    * **corr_z** (`ndarray`):
        The correlation  matrix (:math:`\mathbf{C_Z}`) of the standard normal random vector **Z** .

    **Attributes:**

    * **samples_u** (`ndarray`):
        Uncorrelated standard normal vector of shape ``(nsamples, dimension)``.

    * **H** (`ndarray`):
        The lower diagonal matrix resulting from the Cholesky decomposition of the correlation  matrix
        (:math:`\mathbf{C_Z}`).

    """

    @beartype
    def __init__(self, samples_z: np.ndarray, corr_z: np.ndarray):
        self.samples_z = samples_z
        self.corr_z = corr_z
        self.H = cholesky(self.corr_z, lower=True)
        self.samples_u = np.linalg.solve(self.H, samples_z.T.squeeze()).T

import numpy as np
from beartype import beartype
from scipy.linalg import cholesky


class Correlate:
    """
    A class to induce correlation to standard normal random variables.

    **Inputs:**

    * **samples_u** (`ndarray`):
        Uncorrelated  standard normal vector of shape ``(nsamples, dimension)``.

    * **corr_z** (`ndarray`):
        The correlation  matrix (:math:`\mathbf{C_Z}`) of the standard normal random vector **Z** .

    **Attributes:**

    * **samples_z** (`ndarray`):
        Correlated standard normal vector of shape ``(nsamples, dimension)``.

    * **H** (`ndarray`):
        The lower diagonal matrix resulting from the Cholesky decomposition of the correlation  matrix
        (:math:`\mathbf{C_Z}`).

    """
    @beartype
    def __init__(self,
                 samples_u: np.ndarray,
                 corr_z: np.ndarray):
        self.samples_y = samples_u
        self.corr_z = corr_z
        self.H = cholesky(self.corr_z, lower=True)
        self.samples_z = (self.H @ samples_u.T).T

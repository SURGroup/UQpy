import numpy as np


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

    def __init__(self, samples_u=None, corr_z=None):

        if samples_u is None:
            raise ValueError("UQpy: An  array of samples must be provided.")
        if corr_z is None:
            raise ValueError("UQpy: A correlation matrix must be provided.")

        self.samples_y = samples_u
        self.corr_z = corr_z
        from scipy.linalg import cholesky
        self.H = cholesky(self.corr_z, lower=True)
        self.samples_z = (self.H @ samples_u.T).T

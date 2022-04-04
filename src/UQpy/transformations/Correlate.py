import numpy as np
from beartype import beartype
from scipy.linalg import cholesky

from UQpy.utilities.ValidationTypes import NumpyFloatArray


class Correlate:

    @beartype
    def __init__(self, samples_u: np.ndarray, corr_z: np.ndarray):
        """
        A class to induce correlation to standard normal random variables.

        :param samples_u: Uncorrelated  standard normal vector of shape ``(nsamples, n_dimensions)``.
        :param corr_z: The correlation  matrix (:math:`\mathbf{C_Z}`) of the standard normal random vector **Z** .
        """
        self.samples_y = samples_u
        self.corr_z = corr_z
        self.H: NumpyFloatArray = cholesky(self.corr_z, lower=True)
        """The lower diagonal matrix resulting from the Cholesky decomposition of the correlation  matrix
        (:math:`\mathbf{C_Z}`)."""
        self.samples_z: NumpyFloatArray = (self.H @ samples_u.T).T
        """Correlated standard normal vector of shape ``(nsamples, n_dimensions)``."""

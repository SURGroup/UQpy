import numpy as np
from beartype import beartype

from UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion import (
    PolynomialChaosExpansion,
)


class MomentEstimation:
    @beartype
    def __init__(self, pce_surrogate: PolynomialChaosExpansion):
        """
        Class for estimating the moments of the polynomial_chaos surrogate.

        :param pce_surrogate: Object that defines the surrogate model.
        """
        self.pce_surrogate = pce_surrogate

    def get(self):
        """
        Returns the first two moments of the polynomial_chaos surrogate which are directly
        estimated from the polynomial_chaos coefficients.

        :return: Returns the mean and variance.
        """
        if self.pce_surrogate.b is not None:
            mean = self.pce_surrogate.C[0, :] + np.squeeze(self.pce_surrogate.b)
        else:
            mean = self.pce_surrogate.C[0, :]

        variance = np.sum(self.pce_surrogate.C[1:] ** 2, axis=0)

        if self.pce_surrogate.C.ndim == 1 or self.pce_surrogate.C.shape[1] == 1:
            variance = float(variance)
            mean = float(mean)

        return np.round(mean, 4), np.round(variance, 4)

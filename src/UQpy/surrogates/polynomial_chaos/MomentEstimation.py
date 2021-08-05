import numpy as np


class MomentEstimation:
    """
    Class for estimating the moments of the polynomial_chaos surrogate.

    **Inputs:**

    * **surr_object** ('class'):
        Object that defines the surrogate model.

    **Methods:**
    """

    def __init__(self, pce_surrogate):
        self.pce_surrogate = pce_surrogate

    def get(self):
        """
        Returns the first two moments of the polynomial_chaos surrogate which are directly
        estimated from the polynomial_chaos coefficients.

        **Outputs:**

        * **mean, variance** (`tuple`)
            Returns the mean and variance.

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

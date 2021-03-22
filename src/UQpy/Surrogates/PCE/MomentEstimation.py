import numpy as np
class MomentEstimation:
    """
    Class for estimating the moments of the PCE surrogate.

    **Inputs:**

    * **surr_object** ('class'):
        Object that defines the surrogate model.

    **Methods:**
    """

    def __init__(self, surr_object):
        self.surr_object = surr_object

    def get(self):
        """
        Returns the first two moments of the PCE surrogate which are directly
        estimated from the PCE coefficients.

        **Outputs:**

        * **mean, variance** (`tuple`)
            Returns the mean and variance.

        """
        if self.surr_object.b is not None:
            mean = self.surr_object.C[0, :] + np.squeeze(self.surr_object.b)
        else:
            mean = self.surr_object.C[0, :]

        variance = np.sum(self.surr_object.C[1:] ** 2, axis=0)

        if self.surr_object.C.ndim == 1 or self.surr_object.C.shape[1] == 1:
            variance = float(variance)
            mean = float(mean)

        return np.round(mean, 4), np.round(variance, 4)
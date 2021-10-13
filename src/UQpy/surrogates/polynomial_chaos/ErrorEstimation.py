import numpy as np
from beartype import beartype

from UQpy.surrogates.polynomial_chaos.PolynomialChaosExpansion import (
    PolynomialChaosExpansion,
)


class ErrorEstimation:
    """
    Class for estimating the error of a polynomial_chaos surrogate, based on a validation
    dataset.

    **Inputs:**

    * **surr_object** ('class'):
        Object that defines the surrogate model.

    **Methods:**
    """

    @beartype
    def __init__(self, pce_surrogate: PolynomialChaosExpansion):
        self.pce_surrogate = pce_surrogate

    def validation(self, x, y):
        """
        Returns the validation error.

        **Inputs:**

        * **x** (`ndarray`):
            `ndarray` containing the samples of the validation dataset.

        * **y** (`ndarray`):
            `ndarray` containing model evaluations for the validation dataset.

        **Outputs:**

        * **eps_val** (`float`)
            Validation error.

        """
        if y.ndim == 1 or y.shape[1] == 1:
            y = y.reshape(-1, 1)

        y_val = self.pce_surrogate.predict(x)

        n_samples = x.shape[0]
        mu_yval = (1 / n_samples) * np.sum(y, axis=0)
        eps_val = (
            (n_samples - 1)
            / n_samples
            * (
                (np.sum((y - y_val) ** 2, axis=0))
                / (np.sum((y - mu_yval) ** 2, axis=0))
            )
        )

        if y.ndim == 1 or y.shape[1] == 1:
            eps_val = float(eps_val)

        return np.round(eps_val, 7)

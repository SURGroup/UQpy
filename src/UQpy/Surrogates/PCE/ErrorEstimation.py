import numpy as np

class ErrorEstimation:
    """
    Class for estimating the error of a PCE surrogate, based on a validation
    dataset.

    **Inputs:**

    * **surr_object** ('class'):
        Object that defines the surrogate model.

    **Methods:**
    """

    def __init__(self, surr_object):
        self.surr_object = surr_object

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

        y_val = self.surr_object.predict(x)

        n_samples = x.shape[0]
        mu_yval = (1 / n_samples) * np.sum(y, axis=0)
        eps_val = (n_samples - 1) / n_samples * (
                (np.sum((y - y_val) ** 2, axis=0)) / (np.sum((y - mu_yval) ** 2, axis=0)))

        if y.ndim == 1 or y.shape[1] == 1:
            eps_val = float(eps_val)

        return np.round(eps_val, 7)
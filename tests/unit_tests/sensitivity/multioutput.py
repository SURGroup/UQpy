""""
This is the toy example with multiple outputs from [1]_.

References
----------

.. [1]  Gamboa F, Janon A, Klein T, Lagnoux A, others. 
        Sensitivity analysis for multidimensional and functional outputs.
        Electronic journal of statistics 2014; 8(1): 575-603.

"""

import numpy as np


def evaluate(X):

    """

    * **Input:**

    * **X** (`ndarray`):
    Samples from the input distribution.
    Shape: (n_samples, 2)

    * **Output:**

    * **Y** (`ndarray`):
    Model evaluations.
    Shape: (2, n_samples)

    """

    n_samples = X.shape[0]

    output = np.zeros((2, n_samples))

    output[0, :] = X[:, 0] + X[:, 1] + X[:, 0] * X[:, 1]

    output[1, :] = 2 * X[:, 0] + X[:, 1] + 3 * X[:, 0] * X[:, 1]

    return output

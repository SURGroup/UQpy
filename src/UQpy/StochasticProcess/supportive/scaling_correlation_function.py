import numpy as np


def scaling_correlation_function(correlation_function):
    """
    A function to scale a correlation function such that correlation at 0 lag is equal to 1

    ** Input:**

    * **correlation_function** (`list or numpy.array`):

        The correlation function of the signal.

    **Output/Returns:**

    * **scaled_correlation_function** (`list or numpy.array`):

        The scaled correlation functions of the signal.
    """
    scaled_correlation_function = correlation_function / np.max(correlation_function)
    return scaled_correlation_function

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


def inverse_wiener_khinchin_transform(correlation_function, frequency, time):
    """
    A function to transform the autocorrelation function to a power spectrum by the Inverse Wiener Khinchin
    transformation.

    ** Input:**

    * **correlation_function** (`list or numpy.array`):

        The correlation function of the signal.

    * **frequency** (`list or numpy.array`):

        The frequency discretizations of the power spectrum.

    * **time** (`list or numpy.array`):

        The time discretizations of the signal.

    **Output/Returns:**

    * **power_spectrum** (`list or numpy.array`):

        The power spectrum of the signal.
    """
    time_length = time[1] - time[0]
    fac = np.ones(len(time))
    fac[1: len(time) - 1: 2] = 4
    fac[2: len(time) - 2: 2] = 2
    fac = fac * time_length / 3
    power_spectrum = np.zeros(len(frequency))
    for i in range(len(frequency)):
        power_spectrum[i] = 2 / (2 * np.pi) * np.dot(fac, correlation_function * np.cos(time * frequency[i]))
    power_spectrum[power_spectrum < 0] = 0
    return power_spectrum


def wiener_khinchin_transform(power_spectrum, frequency, time):
    """
    A function to transform the power spectrum to a correlation function by the Wiener Khinchin transformation

    ** Input:**

    * **power_spectrum** (`list or numpy.array`):

        The power spectrum of the signal.

    * **frequency** (`list or numpy.array`):

        The frequency discretizations of the power spectrum.

    * **time** (`list or numpy.array`):

        The time discretizations of the signal.

    **Output/Returns:**

    * **correlation_function** (`list or numpy.array`):

        The correlation function of the signal.

    """
    frequency_interval = frequency[1] - frequency[0]
    fac = np.ones(len(frequency))
    fac[1: len(frequency) - 1: 2] = 4
    fac[2: len(frequency) - 2: 2] = 2
    fac = fac * frequency_interval / 3
    correlation_function = np.zeros(len(time))
    for i in range(len(time)):
        correlation_function[i] = 2 * np.dot(fac, power_spectrum * np.cos(frequency * time[i]))
    return correlation_function

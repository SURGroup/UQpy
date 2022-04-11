import numpy as np


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
    fac[1 : len(frequency) - 1 : 2] = 4
    fac[2 : len(frequency) - 2 : 2] = 2
    fac = fac * frequency_interval / 3
    correlation_function = np.zeros(len(time))
    for i in range(len(time)):
        correlation_function[i] = 2 * np.dot(
            fac, power_spectrum * np.cos(frequency * time[i])
        )
    return correlation_function

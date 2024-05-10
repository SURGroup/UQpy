import numpy as np


def wiener_khinchin_transform(
    power_spectrum: np.ndarray, frequency: np.ndarray, time: np.ndarray
) -> np.ndarray:
    """Transform the power spectrum to a correlation function by the Wiener Khinchin transformation
    ToDo: one the fft works just delete this
    :param power_spectrum: The power spectrum of the signal.
    :param frequency: The frequency discretizations of the power spectrum.
    :param time: The time discretizations of the signal.
    :return: The correlation function of the signal.
    """
    fourier = np.fft.rfft(power_spectrum)
    if np.isnan(fourier).any():
        raise ValueError("wiener_khinchin_transform compute NaN")
    return np.real(fourier)
    # frequency_interval = frequency[1] - frequency[0]
    # fac = np.ones(len(frequency))
    # fac[1: len(frequency) - 1: 2] = 4
    # fac[2: len(frequency) - 2: 2] = 2
    # fac = fac * frequency_interval / 3
    # correlation_function = np.zeros(len(time))
    # for i in range(len(time)):
    #     correlation_function[i] = 2 * np.dot(
    #         fac, power_spectrum * np.cos(frequency * time[i])
    #     )
    # return correlation_function

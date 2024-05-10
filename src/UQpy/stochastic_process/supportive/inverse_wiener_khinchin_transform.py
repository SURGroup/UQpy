import numpy as np


def inverse_wiener_khinchin_transform(
    correlation_function: np.ndarray, frequency: np.ndarray, time: np.ndarray
) -> np.ndarray:
    """Transform the autocorrelation function to a power spectrum by the Inverse Wiener Khinchin transformation.
    ToDo: once the irfft works just delete this
    :param correlation_function: The correlation function of the signal.
    :param frequency: The frequency discretizations of the power spectrum.
    :param time: The time discretizations of the signal.
    :return: The power spectrum of the signal.
    """
    inverse_fourier = np.fft.irfft(correlation_function)
    if np.isnan(inverse_fourier).any():
        raise ValueError("inverse_wiener_khinchin_transform computed NaN")
    return np.real(inverse_fourier)
    # time_length = time[1] - time[0]
    # fac = np.ones(len(time))
    # fac[1 : len(time) - 1 : 2] = 4
    # fac[2 : len(time) - 2 : 2] = 2
    # fac = fac * time_length / 3
    # power_spectrum = np.zeros(len(frequency))
    # for i in range(len(frequency)):
    #     power_spectrum[i] = (
    #         2
    #         / (2 * np.pi)
    #         * np.dot(fac, correlation_function * np.cos(time * frequency[i]))
    #     )
    # power_spectrum[power_spectrum < 0] = 0
    # return power_spectrum

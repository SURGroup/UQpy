import numpy as np
import matplotlib.pyplot as plt
from UQpy.stochastic_process import SpectralRepresentation

plt.style.use("ggplot")


# Sample parameters
n_samples = 1
n_variables = 1
n_dimensions = 1
# Spectral Representation Parameters
n_times = 256
n_frequencies = 128
max_frequency = 12.8
frequency_interval = max_frequency / n_frequencies
frequencies = np.linspace(
    0, (n_frequencies - 1) * frequency_interval, num=n_frequencies
)
max_time = 2 * np.pi / frequency_interval
time_interval = max_time / n_times

"""Test phi is zere"""
phi = np.zeros(shape=(1, n_frequencies))
phi[0, 5] = 3
"""Test constant phi"""
# phi = np.ones(shape=(1, n_frequencies)) * np.pi / 4
"""Test random phase angles"""
# phi = np.random.uniform(low=0, high=2 * np.pi, size=(1, n_frequencies))

"""Test simple power spectrum with contribution at only two frequencies"""
power_spectrum = np.zeros(shape=n_frequencies)
power_spectrum[1] = 1  # Has power at frequency 1 * frequency_interval = 0.1
power_spectrum[5] = 1  # Has power at frequency 100 * frequency_interval = 0.5
"""Test more interesting power spectrum"""
# power_spectrum = (125 / 4) * (frequencies ** 2) * np.exp(-5 * frequencies)


def srm_object():
    srm = SpectralRepresentation(
        power_spectrum, time_interval, frequency_interval, n_times, n_frequencies
    )
    srm.run(n_samples=n_samples, phase_angles=phi)
    return srm


def sum_of_cosines(
    power_spectrum, phi, time_interval, frequency_interval, n_times, n_frequencies
):
    """Computes the stochastic process as sum of cosines"""
    time = np.linspace(0, (n_times - 1) * time_interval, num=n_times)
    total = np.zeros(n_times)
    for i in range(n_frequencies):
        w_i = frequency_interval * i
        coefficient = 2 * np.sqrt(power_spectrum[i] * frequency_interval)
        term = coefficient * np.cos(
            -(w_i * time) + phi[0, i]
        )  # Put a minus sign in the cos to fix it
        total += term
    return total


if __name__ == "__main__":
    srm = srm_object()

    # Compute the Spectral Representation as a direct sum of cosines
    time = np.linspace(0, (n_times - 1) * time_interval, num=n_times)
    cosines = sum_of_cosines(
        power_spectrum,
        srm.phase_angles,
        time_interval,
        frequency_interval,
        n_times,
        n_frequencies,
    )
    print(srm.n_variables)
    print(cosines.shape)
    print(srm.phase_angles.shape)
    print(srm.samples.shape)
    # print(srm.phi)
    print(
        "all is close:", all(np.isclose(np.squeeze(srm.samples), np.squeeze(cosines)))
    )
    fig, ax = plt.subplots()
    ax.plot(time, srm.samples[0, 0, :], label="SRM FFT")
    ax.plot(time, cosines, linestyle=":", label="Sum of Cosines")
    ax.legend()
    plt.show()

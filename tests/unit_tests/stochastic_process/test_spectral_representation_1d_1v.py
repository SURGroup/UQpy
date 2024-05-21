import pytest
import numpy as np
from UQpy.stochastic_process import SpectralRepresentation


# Sampling parameters
n_variables = 1
n_dimensions = 1
# Spectrum parameters
n_times = 256
n_frequencies = 128
max_frequency = 12.8
frequency_interval = max_frequency / n_frequencies
frequencies = np.linspace(
    0, (n_frequencies - 1) * frequency_interval, num=n_frequencies
)
max_time = 2 * np.pi / frequency_interval
time_interval = max_time / n_times


def sum_of_cosines(phase_angles, power_spectrum):
    """Computes the stochastic process as sum of cosines"""
    n_samples = phase_angles.shape[0]
    samples_shape = np.append(np.array([n_samples, n_variables]), n_times)
    samples = np.zeros(samples_shape)
    time = np.linspace(0, (n_times - 1) * time_interval, num=n_times)
    for i in range(n_samples):
        for j in range(n_frequencies):
            w_j = frequency_interval * j
            coefficient = 2 * np.sqrt(power_spectrum[j] * frequency_interval)
            term = coefficient * np.cos((w_j * time) + phase_angles[i, 0, j])
            samples[i, :] += term
    return samples


@pytest.mark.parametrize("n_samples", [1, 2, 10, 100])
def test_1d_1v_sample_shape(n_samples):
    power_spectrum = np.ones_like(frequencies)
    srm = SpectralRepresentation(
        power_spectrum,
        time_interval,
        frequency_interval,
        n_times,
        n_frequencies,
        n_samples=n_samples,
    )
    assert srm.samples.shape == (n_samples, n_variables, n_times)
    assert srm.phase_angles.shape == (n_samples, n_variables, n_frequencies)


@pytest.mark.parametrize(
    "phase_angles",
    [
        np.zeros(shape=(1, 1, n_frequencies)),
        np.ones(shape=(1, 1, n_frequencies)) * np.pi / 4,
        np.random.uniform(0, 2 * np.pi, size=(1, 1, n_frequencies)),
    ],
)
@pytest.mark.parametrize(
    "power_spectrum",
    [np.ones_like(frequencies), (frequencies**2) * np.exp(-frequencies)],
)
def test_1d_1v_sample_accuracy(phase_angles, power_spectrum):
    """Compare the samples generated via FFT in the SRM to the sum of cosines"""
    srm = SpectralRepresentation(
        power_spectrum, time_interval, frequency_interval, n_times, n_frequencies
    )
    srm.run(n_samples=1, phase_angles=phase_angles)
    cosines = sum_of_cosines(srm.phase_angles, srm.power_spectrum)
    assert np.allclose(srm.samples, cosines)


def test_run_on_init():
    """Test run method called via n_samples on init"""
    n_samples = 1
    power_spectrum = np.ones_like(frequencies)
    srm = SpectralRepresentation(
        power_spectrum,
        time_interval,
        frequency_interval,
        n_times,
        n_frequencies,
        n_samples=n_samples,
        random_state=123,
    )
    cosines = sum_of_cosines(srm.phase_angles, srm.power_spectrum)
    assert srm.samples.shape == (n_samples, n_variables, n_times)
    assert np.allclose(srm.samples, cosines)


def test_aliasing_raises_error():
    """When the time interval is too large, raise RuntimeError"""
    power_spectrum = np.zeros_like(frequencies)
    with pytest.raises(RuntimeError):
        srm = SpectralRepresentation(
            power_spectrum, 100, frequency_interval, n_times, n_frequencies
        )

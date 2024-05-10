import pytest
import numpy as np
from UQpy.stochastic_process import SpectralRepresentation


# Sample parameters
n_variables = 1
n_dimensions = 2
# Spectral Representation Parameters
n_dimension_intervals = np.array([256, 256])
n_frequency_intervals = np.array([128, 128])
max_frequency = np.array([12.8, 6.4])

frequency_interval = max_frequency / n_frequency_intervals
frequency_vectors = [
    np.linspace(0, (n_frequency_intervals[i] - 1) * frequency_interval[i], num=n_frequency_intervals[i])
    for i in range(n_dimensions)
]
x_frequencies, y_frequencies = np.meshgrid(*frequency_vectors, indexing="ij")

max_time = 2 * np.pi / frequency_interval
time_interval = max_time / n_dimension_intervals


def sum_of_cosines(phi, power_spectrum):
    """Stochastic process defined using sum of cosines from Eq 44 of
    Simulation of multidimensional Gaussian stochastic fields by spectral representation (1996)
    """
    x_vector = np.linspace(
        0, (n_dimension_intervals[0] - 1) * time_interval[0], num=n_frequency_intervals[0]
    )
    y_vector = np.linspace(
        0, (n_dimension_intervals[1] - 1) * time_interval[1], num=n_frequency_intervals[1]
    )
    x_array, y_array = np.meshgrid(x_vector, y_vector)

    total = np.full(phi.shape, np.nan)
    for i in range(n_frequency_intervals[0]):
        for j in range(n_frequency_intervals[1]):
            kappa_1 = i * frequency_interval[0]
            kappa_2 = j * frequency_interval[1]
            coefficient_1 = np.sqrt(
                2 * power_spectrum(kappa_1, kappa_2) * np.prod(frequency_interval)
            )
            coefficient_2 = np.sqrt(
                2 * power_spectrum(kappa_1, -kappa_2) * np.prod(frequency_interval)
            )
            input_1 = (kappa_1 * x_array) + (kappa_2 * y_array) + phi[0, i, j]
            input_2 = (kappa_1 * x_array) - (kappa_2 * y_array) + phi[1, i, j]  #  FixMe: The phi indexing  is wrong
            term = coefficient_1 * np.cos(input_1) + coefficient_2 * np.cos(input_2)
            total += term
    return total


@pytest.mark.parametrize("n_samples", [1, 2, 10, 100])
def test_2d_1v_shape(n_samples):
    srm = SpectralRepresentation(
        np.ones_like(x_frequencies),
        time_interval,
        frequency_interval,
        n_dimension_intervals,
        n_frequency_intervals,
        n_samples=n_samples,
    )
    assert srm.samples.shape == (
        n_samples,
        n_variables,
        n_dimension_intervals[0],
        n_dimension_intervals[1],
    )
    assert srm.phase_angles.shape == (
        n_samples,
        n_frequency_intervals[0],
        n_frequency_intervals[1],
    )


@pytest.mark.parametrize(
    "phase_angles",
    [
        np.zeros(shape=(1, *n_frequency_intervals)),
        np.ones(shape=(1, *n_frequency_intervals)) * np.pi / 4,
        np.random.uniform(0, 2 * np.pi, size=(1, *n_frequency_intervals)),
    ],
)
@pytest.mark.parametrize(
    "power_spectrum",
    [
        lambda x, y: 1,
        lambda x, y: np.exp(-abs(x + y)),
        lambda x, y: ((x + y) ** 2) * np.exp(-abs(x + y)),
    ],
)
def test_nd_1v_values(phase_angles, power_spectrum):
    spectrum = np.full(n_frequency_intervals, np.nan)
    for i in range(n_frequency_intervals[0]):
        for j in range(n_frequency_intervals[1]):
            w_1 = i * frequency_interval[0]
            w_2 = j * frequency_interval[1]
            spectrum[i, j] = power_spectrum(w_1, w_2)

    srm = SpectralRepresentation(
        spectrum,
        time_interval,
        frequency_interval,
        n_dimension_intervals,
        n_frequency_intervals,
        n_samples=1,
        phase_angles=None,
    )
    samples = srm.samples[0, 0, :, :]
    cosines = sum_of_cosines(phase_angles, power_spectrum)
    assert np.allclose(samples, cosines)


def test_aliasing_raises_error():
    """When the time interval is too large, raise RuntimeError"""
    power_spectrum = np.ones_like(x_frequencies)
    with pytest.raises(RuntimeError):
        srm = SpectralRepresentation(
            power_spectrum,
            np.array([100, 0.001]),
            frequency_interval,
            n_dimension_intervals,
            n_frequency_intervals,
        )

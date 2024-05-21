import pytest
import numpy as np
from UQpy.stochastic_process import SpectralRepresentation


# Sample parameters
n_variables = 1
n_dimensions = 2
# Spectral Representation Parameters
n_dimension_intervals = np.array([256, 512])
n_frequency_intervals = np.array([128, 256])
max_frequency = np.array([5.12, 10.24])

# Great test case 97 / 131
# n_dimension_intervals = np.array([256, 512])
# n_frequency_intervals = np.array([128, 256])
# max_frequency = np.array([2.56, 5.12])

frequency_interval = max_frequency / n_frequency_intervals
frequency_vectors = [
    np.linspace(
        0,
        (n_frequency_intervals[i] - 1) * frequency_interval[i],
        num=n_frequency_intervals[i],
    )
    for i in range(n_dimensions)
]
x_frequencies, y_frequencies = np.meshgrid(*frequency_vectors, indexing="ij")

max_time = 2 * np.pi / frequency_interval
time_interval = max_time / n_dimension_intervals


def sum_of_cosines(phase_angles, power_spectrum):
    """Stochastic process defined used sum of cosines from equation 60 of Shinozuka 1988

    Reference
    ---------
    M. Shinozuka, G. Deodatis,
    Stochastic process models for earthquake ground motion,
    Probabilistic Engineering Mechanics, Volume 3, Issue 3, 1988, Pages 114-123,
    ISSN 0266-8920, https://doi.org/10.1016/0266-8920(88)90023-9.

    :param phase_angles: Realizations of i.i.d Uniform(0, 2pi) random variables
    :param power_spectrum: Power spectrum :math:`S(\\omega_1, \\omega_2)`
    :return: Samples
    """
    n_samples = phase_angles.shape[0]
    samples_shape = np.append(np.array([n_samples, n_variables]), n_dimension_intervals)
    samples = np.zeros(samples_shape)

    x_vector = np.linspace(
        0,
        (n_dimension_intervals[1] - 1) * time_interval[1],
        num=n_dimension_intervals[1],
    )
    y_vector = np.linspace(
        0,
        (n_dimension_intervals[0] - 1) * time_interval[0],
        num=n_dimension_intervals[0],
    )
    x, y = np.meshgrid(x_vector, y_vector)

    for i in range(n_samples):
        for j in range(n_frequency_intervals[0]):
            for k in range(n_frequency_intervals[1]):
                coefficient = np.sqrt(power_spectrum[j, k])
                kappa_1 = j * frequency_interval[0]
                kappa_2 = k * frequency_interval[1]
                cos_input = (kappa_1 * x) + (kappa_2 * y) + phase_angles[i, 0, j, k]
                term = coefficient * np.cos(cos_input)
                samples[i, 0, :, :] += term
    return 2 * np.sqrt(np.prod(frequency_interval)) * samples


@pytest.mark.parametrize("n_samples", [1, 2, 10, 100])
def test_2d_1v_shape(n_samples):
    power_spectrum = np.ones_like(x_frequencies)
    srm = SpectralRepresentation(
        power_spectrum,
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
        n_variables,
        n_frequency_intervals[0],
        n_frequency_intervals[1],
    )


@pytest.mark.parametrize(
    "phase_angles",
    [
        np.zeros(shape=(1, n_variables, *n_frequency_intervals)),
        np.full((1, n_variables, *n_frequency_intervals), np.pi / 4),
        # np.random.uniform(0, 2 * np.pi, size=(1, n_variables, *n_frequency_intervals)),
    ],
)
@pytest.mark.parametrize(
    "compute_power_spectrum",
    [
        # lambda x, y: np.ones_like(x),
        lambda x, y: np.exp(-((x + y) ** 2)),
        # lambda x, y: ((x + y) ** 2) * np.exp(-abs(x + y)),
    ],
)
def test_2d_1v_values(phase_angles, compute_power_spectrum):
    """Compare the SRM samples to the sum of cosines formula from equation 60 of Shinozuka, Deodatis 1988

    :param phase_angles: Realizations of i.i.d Uniform(0, 2pi) random variables
    :param compute_power_spectrum: Callable function defining the power spectrum :math:`S(\\omega_1, \\omega_2)`
    """
    power_spectrum = compute_power_spectrum(x_frequencies, y_frequencies)
    srm = SpectralRepresentation(
        power_spectrum,
        time_interval,
        frequency_interval,
        n_dimension_intervals,
        n_frequency_intervals,
        n_samples=1,
        phase_angles=phase_angles,
    )
    cosines = sum_of_cosines(srm.phase_angles, srm.power_spectrum)
    assert np.allclose(srm.samples, cosines, rtol=0.1)


def test_aliasing_raises_error():
    """When the time interval is too large, raise RuntimeError"""
    power_spectrum = np.ones_like(x_frequencies)
    with pytest.raises(RuntimeError):
        SpectralRepresentation(
            power_spectrum,
            np.array([100, 0.001]),
            frequency_interval,
            n_dimension_intervals,
            n_frequency_intervals,
        )


def test_user_input_raises_error():
    """If phase_angles is defined but n_samples is not, raise a runtime error"""
    power_spectrum = np.ones_like(x_frequencies)
    with pytest.raises(RuntimeError):
        SpectralRepresentation(
            power_spectrum,
            time_interval,
            frequency_interval,
            n_dimension_intervals,
            n_frequency_intervals,
            n_samples=None,
            phase_angles=np.array([1]),
        )

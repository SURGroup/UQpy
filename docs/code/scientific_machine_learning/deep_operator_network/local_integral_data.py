import torch
import numpy as np
from UQpy.stochastic_process import SpectralRepresentation


def srm_2d_samples(n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a 2D Gaussian process using the Spectral Representation Method

    :param n_samples: Number of samples. Each sample is one row
    :return: time, samples
    """
    n_dimension = 2
    max_time = np.array([1.0, 2.0])
    n_time = np.array([50, 100])
    max_frequency = np.array([8 * np.pi, 4 * np.pi])
    n_frequency = np.array([64, 32])

    delta_time = max_time / n_time
    delta_frequency = max_frequency / n_frequency

    frequency_vectors = [
        np.linspace(0, max_frequency[i] - delta_frequency[i], n=n_frequency[i])
        for i in range(n_dimension)
    ]
    frequency = np.array(np.meshgrid(*frequency_vectors, indexing="ij"))
    time_vectors = [
        np.linspace(0, max_time[i] - delta_time[i], num=n_time[i])
        for i in range(n_dimension)
    ]
    time = np.array(np.meshgrid(*time_vectors, indexing="ij"))

    power_spectrum = np.exp(-2 * np.linalg.norm(frequency, axis=0))
    srm = SpectralRepresentation(
        n_samples, power_spectrum, delta_time, delta_frequency, n_time, n_frequency
    )
    return (
        torch.tensor(time, dtype=torch.float).reshape(-1, 2),
        torch.tensor(srm.samples, dtype=torch.float),
    )

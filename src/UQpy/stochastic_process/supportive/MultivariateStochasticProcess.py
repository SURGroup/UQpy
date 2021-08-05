import logging

import numpy as np

class MultivariateStochasticProcess():

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_samples(self):
        self.logger.info('UQpy: Stochastic Process: Starting simulation of multi-variate Stochastic Processes.')
        self.logger.info('UQpy: Stochastic Process: The number of variables is :', self.number_of_variables)
        self.logger.info('UQpy: Stochastic Process: The number of dimensions is :', self.number_of_dimensions)
        phi = np.random.uniform(size=np.append(self.samples_number, np.append(
            np.ones(self.number_of_dimensions, dtype=np.int32) * self.number_frequency_intervals,
            self.number_of_variables))) * 2 * np.pi
        samples = self._simulate_multi(phi)

        return samples

    def _simulate_multi(self, phi):
        power_spectrum = np.einsum('ij...->...ij', self.power_spectrum)
        coefficient = np.sqrt(2 ** (self.number_of_dimensions + 1)) * np.sqrt(np.prod(self.frequency_interval))
        u, s, v = np.linalg.svd(power_spectrum)
        power_spectrum_decomposed = np.einsum('...ij,...j->...ij', u, np.sqrt(s))
        fourier_coefficient = coefficient * np.einsum('...ij,trials_number...j -> trials_number...i',
                                                      power_spectrum_decomposed, np.exp(phi * 1.0j))
        fourier_coefficient[np.isnan(fourier_coefficient)] = 0
        samples = np.real(np.fft.fftn(fourier_coefficient, s=self.number_time_intervals,
                                      axes=tuple(np.arange(1, 1 + self.number_of_dimensions))))
        samples = np.einsum('n...m->nm...', samples)
        return samples

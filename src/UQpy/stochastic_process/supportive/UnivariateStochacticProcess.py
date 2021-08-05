import logging

import numpy as np

class UnivariateStochasticProcess():

    def __init__(self):
        self.logger=logging.getLogger(__name__)

    def calculate_samples(self):
        self.logger.info('UQpy: Stochastic Process: Starting simulation of uni-variate Stochastic Processes.')
        self.logger.info('UQpy: The number of dimensions is :', self.number_of_dimensions)
        phi = np.random.uniform(
            size=np.append(self.samples_number, np.ones(self.number_of_dimensions, dtype=np.int32)
                           * self.number_frequency_intervals)) * 2 * np.pi
        samples = self._simulate_uni(phi)
        return samples

    def _simulate_uni(self, phi):
        fourier_coefficient = np.exp(phi * 1.0j) * np.sqrt(
            2 ** (self.number_of_dimensions + 1) * self.power_spectrum * np.prod(self.frequency_interval))
        samples = np.fft.fftn(fourier_coefficient, self.number_time_intervals)
        samples = np.real(samples)
        samples = samples[:, np.newaxis]
        return samples

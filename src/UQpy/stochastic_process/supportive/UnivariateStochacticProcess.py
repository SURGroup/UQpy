import logging

import numpy as np


class UnivariateStochasticProcess:

    def __init__(self,
                 number_of_variables,
                 number_of_dimensions,
                 frequency_intervals_number,
                 time_intervals_number,
                 power_spectrum,
                 frequency_interval):
        self.time_intervals_number = time_intervals_number
        self.frequency_interval = frequency_interval
        self.power_spectrum = power_spectrum
        self.frequency_intervals_number = frequency_intervals_number
        self.number_of_dimensions = number_of_dimensions
        self.number_of_variables = number_of_variables
        self.logger = logging.getLogger(__name__)

    def calculate_samples(self, samples_number):
        self.logger.info('UQpy: Stochastic Process: Starting simulation of uni-variate Stochastic Processes.')
        self.logger.info('UQpy: The number of dimensions is :', self.number_of_dimensions)
        phi = np.random.uniform(
            size=np.append(samples_number, np.ones(self.number_of_dimensions, dtype=np.int32)
                           * self.frequency_intervals_number)) * 2 * np.pi
        samples = self._simulate_uni(phi)
        return samples

    def _simulate_uni(self, phi):
        fourier_coefficient = np.exp(phi * 1.0j) * np.sqrt(
            2 ** (self.number_of_dimensions + 1) * self.power_spectrum * np.prod(self.frequency_interval))
        samples = np.fft.fftn(fourier_coefficient, self.time_intervals_number)
        samples = np.real(samples)
        samples = samples[:, np.newaxis]
        return samples

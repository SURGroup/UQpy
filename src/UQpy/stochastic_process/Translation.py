import itertools

import numpy as np
from scipy.stats import norm

from UQpy.distributions.baseclass import Distribution
from UQpy.utilities import *
from UQpy.stochastic_process.supportive import (
    inverse_wiener_khinchin_transform,
    wiener_khinchin_transform,
    scaling_correlation_function,
)


class Translation:
    def __init__(
            self,
            distributions: Distribution,
            time_interval: Union[list, np.ndarray, float],
            frequency_interval: Union[list, np.ndarray, float],
            n_time_intervals: Union[list, np.ndarray, float],
            n_frequency_intervals: Union[list, np.ndarray, float],
            power_spectrum_gaussian: np.ndarray = None,
            correlation_function_gaussian: np.ndarray = None,
            samples_gaussian: np.ndarray = None,
    ):
        """
        A class to translate Gaussian Stochastic Processes to non-Gaussian Stochastic Processes

        :param distributions: An instance of the UQpy :class:`.Distribution` class defining the marginal distribution to
         which the Gaussian stochastic process should be translated to.
        :param time_interval: The value of time discretization.
        :param frequency_interval: The value of frequency discretization.
        :param n_time_intervals: The number of time discretizations.
        :param n_frequency_intervals: The number of frequency discretizations.
        :param power_spectrum_gaussian: The power spectrum of the gaussian stochastic process to be translated.
         `power_spectrum_gaussian` must be of size :code:`(n_frequency_intervals)`.
        :param correlation_function_gaussian: The auto correlation function of the Gaussian stochastic process to be
         translated. Either the power spectrum or the auto correlation function of the gaussian stochastic process needs
         to be defined. `correlation_function_gaussian` must be of size :code:`(n_time_intervals)`.
        :param samples_gaussian: Samples of Gaussian stochastic process to be translated.
         `samples_gaussian` is optional. If no samples are passed, the :class:`.Translation` class will compute the
         correlation distortion.
        """
        self.samples_non_gaussian = None
        self.samples_gaussian = None
        """Translated non-Gaussian stochastic process from Gaussian samples."""
        self.samples_shape = None
        self.distributions = distributions
        self.time_interval = time_interval
        self.frequency_interval = frequency_interval
        self.n_time_intervals = n_time_intervals
        self.n_frequency_intervals = n_frequency_intervals
        self.correlation_function_non_gaussian: callable = None
        """The correlation function of the translated non-Gaussian stochastic processes obtained by distorting the 
        Gaussian correlation function."""
        self.scaled_correlation_function_non_gaussian: callable = None
        """This obtained by scaling the correlation function of the non-Gaussian stochastic processes to make the
        correlation at '0' lag to be 1"""
        if correlation_function_gaussian is None and power_spectrum_gaussian is None:
            print("Either the Power Spectrum or the Autocorrelation function should be specified")
        if correlation_function_gaussian is None:
            self.power_spectrum_gaussian = power_spectrum_gaussian
            self.correlation_function_gaussian = wiener_khinchin_transform(
                power_spectrum_gaussian,
                np.arange(0, self.n_frequency_intervals) * self.frequency_interval,
                np.arange(0, self.n_time_intervals) * self.time_interval,)
        elif power_spectrum_gaussian is None:
            self.correlation_function_gaussian = correlation_function_gaussian
            self.power_spectrum_gaussian = inverse_wiener_khinchin_transform(
                correlation_function_gaussian,
                np.arange(0, self.n_frequency_intervals) * self.frequency_interval,
                np.arange(0, self.n_time_intervals) * self.time_interval,)
        self.shape = self.correlation_function_gaussian.shape
        self.dim = len(self.correlation_function_gaussian.shape)
        if samples_gaussian is not None:
            self.run(samples_gaussian)

        (self.correlation_function_non_gaussian, self.scaled_correlation_function_non_gaussian,) \
            = self._autocorrelation_distortion()
        self.power_spectrum_non_gaussian: NumpyFloatArray = inverse_wiener_khinchin_transform(
            self.correlation_function_non_gaussian,
            np.arange(0, self.n_frequency_intervals) * self.frequency_interval,
            np.arange(0, self.n_time_intervals) * self.time_interval,)
        """The power spectrum of the translated non-Gaussian stochastic processes."""

    def run(self, samples_gaussian):
        """

        :param samples_gaussian: Samples of Gaussian stochastic process to be translated.
         `samples_gaussian` is optional. If samples are provided at the object initialization then the run method is
         executed automatically. If no samples are passed, the :class:`.Translation` class will compute the
         correlation distortion and the use needs to execute the run method manually in order to compute non-gaussian
         samples.
        """
        self.samples_shape = samples_gaussian.shape
        self.samples_gaussian: NumpyFloatArray = samples_gaussian.flatten()[:, np.newaxis]
        self.samples_non_gaussian = self._translate_gaussian_samples().reshape(self.samples_shape)

    def _translate_gaussian_samples(self):
        standard_deviation = np.sqrt(self.correlation_function_gaussian[0])
        samples_cdf = norm.cdf(self.samples_gaussian, scale=standard_deviation)
        if not hasattr(self.distributions, "icdf"):
            raise AttributeError("UQpy: The marginal dist_object needs to have an inverse cdf defined.")
        non_gaussian_icdf = getattr(self.distributions, "icdf")
        return non_gaussian_icdf(samples_cdf)

    def _autocorrelation_distortion(self):
        correlation_function_gaussian = scaling_correlation_function(self.correlation_function_gaussian)
        correlation_function_gaussian = np.clip(correlation_function_gaussian, -0.999, 0.999)
        correlation_function_non_gaussian = np.zeros_like(correlation_function_gaussian)
        for i in itertools.product(*[range(s) for s in self.shape]):
            correlation_function_non_gaussian[i] = correlation_distortion(
                self.distributions, correlation_function_gaussian[i])
        if hasattr(self.distributions, "moments"):
            non_gaussian_moments = getattr(self.distributions, "moments")()
        else:
            raise AttributeError("UQpy: The marginal dist_object needs to have defined moments.")
        scaled_correlation_function_non_gaussian = correlation_function_non_gaussian * non_gaussian_moments[1] + \
                                                   non_gaussian_moments[0] ** 2
        return correlation_function_non_gaussian, scaled_correlation_function_non_gaussian

import itertools
import numpy as np
from scipy.stats import norm
from UQpy.distributions.baseclass import Distribution
from UQpy.utilities import correlation_distortion
from UQpy.stochastic_process.supportive import (
    inverse_wiener_khinchin_transform,  # ToDo: replace inverse and forward with FFT
    wiener_khinchin_transform
)
from beartype import beartype
from typing import Annotated, Union
from beartype.vale import Is


@beartype
class Translation:
    def __init__(
        self,
        distributions: Annotated[Distribution, Is[lambda d: hasattr(d, "moments")]],
        time_interval: Union[list, np.ndarray, float],
        frequency_interval: Union[list, np.ndarray, float],
        n_time_intervals: Union[list, np.ndarray, float],
        n_frequency_intervals: Union[list, np.ndarray, float],
        power_spectrum_gaussian: np.ndarray = None,
        correlation_function_gaussian: np.ndarray = None,
        samples_gaussian: np.ndarray = None,
    ):
        """A class to translate Gaussian Stochastic Processes to non-Gaussian Stochastic Processes

        :param distributions: An instance of the UQpy :class:`.Distribution` class defining the marginal distribution to
         which the Gaussian stochastic process should be translated to.
        :param time_interval: The value of time discretization.
        :param frequency_interval: The value of frequency discretization.
        :param n_time_intervals: The number of time discretizations.
        :param n_frequency_intervals: The number of frequency discretizations.
        :param power_spectrum_gaussian: The power spectrum of the gaussian stochastic process to be translated.
         Must be of size ``(n_frequency_intervals)``.
         Either ``power_spectrum_gaussian`` or ``correlation_function_gaussian`` must be defined.
        :param correlation_function_gaussian: The auto correlation function of the Gaussian stochastic process to be
         translated. Must be of size ``(n_time_intervals)``.
         Either ``power_spectrum_gaussian`` or ``correlation_function_gaussian`` must be defined.
        :param samples_gaussian: Samples of Gaussian stochastic process to be translated.
         `samples_gaussian` is optional. If no samples are passed, the :class:`.Translation` class will compute the
         correlation distortion.
        """
        self.distributions = distributions
        self.time_interval = time_interval
        self.frequency_interval = frequency_interval
        self.n_time_intervals = n_time_intervals
        self.n_frequency_intervals = n_frequency_intervals
        self.samples_gaussian = samples_gaussian
        if (  # input only power_spectrum_gaussian
            correlation_function_gaussian is None
            and power_spectrum_gaussian is not None
        ):
            self.power_spectrum_gaussian = power_spectrum_gaussian
            self.correlation_function_gaussian = wiener_khinchin_transform(
                power_spectrum_gaussian,
                np.arange(0, self.n_frequency_intervals) * self.frequency_interval,
                np.arange(0, self.n_time_intervals) * self.time_interval,
            )
        elif (  # input only correlation_function_gaussian
            correlation_function_gaussian is not None
            and power_spectrum_gaussian is None
        ):
            self.correlation_function_gaussian = correlation_function_gaussian
            self.power_spectrum_gaussian = inverse_wiener_khinchin_transform(
                correlation_function_gaussian,
                np.arange(0, self.n_frequency_intervals) * self.frequency_interval,
                np.arange(0, self.n_time_intervals) * self.time_interval,
            )
        else:
            raise RuntimeError(
                "UQpy: Exactly one of `correlation_function_gaussian` or `power_spectrum_gaussian` must be provided."
            )

        self.samples_non_gaussian = None
        """Translated non-Gaussian stochastic process from Gaussian samples."""
        self.correlation_function_non_gaussian: np.ndarray = None
        """The correlation function of the translated non-Gaussian stochastic processes obtained by distorting the 
        Gaussian correlation function."""
        self.scaled_correlation_function_non_gaussian: np.ndarray = None
        """This obtained by scaling the correlation function of the non-Gaussian stochastic processes to make the
        correlation at '0' lag to be 1"""
        (
            self.correlation_function_non_gaussian,
            self.scaled_correlation_function_non_gaussian,
        ) = self._autocorrelation_distortion()
        self.power_spectrum_non_gaussian: np.ndarray = (
            inverse_wiener_khinchin_transform(
                self.correlation_function_non_gaussian,
                np.arange(0, self.n_frequency_intervals) * self.frequency_interval,
                np.arange(0, self.n_time_intervals) * self.time_interval,
            )
        )
        """The power spectrum :math:`S_{NG}(\omega)` of the translated non-Gaussian stochastic processes."""

        if self.samples_gaussian:
            self.run(self.samples_gaussian)

    def run(self, samples_gaussian: np.ndarray):
        """Run the forward Translation Approximation Method to convert Gaussian samples into non-Gaussian samples.

        :param samples_gaussian: Samples of Gaussian stochastic process to be translated.
         If ``samples_gaussian`` is provided at initialization, then the run method is executed automatically.
         If no samples are passed at initialization, the :class:`.Translation` class will compute the correlation
         distortion and the use needs to execute the run method manually in order to compute non-gaussian samples.
        """
        self.samples_gaussian = samples_gaussian.flatten()[:, np.newaxis]
        self.samples_non_gaussian = self._translate_gaussian_samples().reshape(
            samples_gaussian.shape
        )

    def _translate_gaussian_samples(self):
        """Translate zero mean gaussian samples to non-gaussian samples using the inverse CDF of the distribution"""
        standard_deviation = np.sqrt(self.correlation_function_gaussian[0])
        samples_cdf = norm.cdf(self.samples_gaussian, scale=standard_deviation)
        if hasattr(self.distributions, "icdf"):  # ToDo: can I move this to the beartype
            return self.distributions.icdf(samples_cdf)
        else:
            raise AttributeError(
                "UQpy: The marginal dist_object needs to have an inverse cdf defined."
            )

    def _autocorrelation_distortion(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the non-Gaussian correlation function from the Gaussian correlation and target distribution"""
        correlation_function_gaussian = self.correlation_function_gaussian / np.max(
            self.correlation_function_gaussian
        )
        correlation_function_gaussian = np.clip(
            correlation_function_gaussian, -0.999, 0.999
        )
        correlation_function_non_gaussian = np.zeros_like(correlation_function_gaussian)
        for i in itertools.product(*[range(s) for s in self.shape]):
            correlation_function_non_gaussian[i] = correlation_distortion(
                self.distributions, correlation_function_gaussian[i]
            )
        if hasattr(self.distributions, "moments"):  # ToDo: can I move this to beartype
            non_gaussian_mean, non_gaussian_variance = self.distributions.moments(
                moments2return="mv"
            )
        else:
            raise AttributeError(
                "UQpy: The input `distributions` must have moments defined by `distribution.moments()`."
            )
        scaled_correlation_function_non_gaussian = (
            correlation_function_non_gaussian * non_gaussian_mean
            + non_gaussian_variance**2
        )  # ToDo: can i move this calculation to the __init__ and just have this return one thing
        return (
            correlation_function_non_gaussian,
            scaled_correlation_function_non_gaussian,
        )

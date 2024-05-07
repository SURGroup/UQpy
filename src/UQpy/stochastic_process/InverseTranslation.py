import numpy as np
from UQpy.distributions import Distribution, Normal
from UQpy.utilities import *
from UQpy.utilities.ValidationTypes import PositiveFloat
from UQpy.stochastic_process.supportive import (
    inverse_wiener_khinchin_transform,
    wiener_khinchin_transform,
)
from beartype import beartype


@beartype
class InverseTranslation:
    def __init__(
        self,
        distributions: Distribution,
        time_interval: Union[list, np.ndarray],
        frequency_interval: Union[list, np.ndarray],
        n_time_intervals: Union[list, np.ndarray],
        n_frequency_intervals: Union[list, np.ndarray],
        target_correlation_function_non_gaussian: Union[list, np.ndarray] = None,
        target_power_spectrum_non_gaussian: Union[list, np.ndarray] = None,
        samples_non_gaussian: Union[list, np.ndarray] = None,
        percentage_error: PositiveFloat = 5.0,
    ):
        """A class to perform Iterative Translation Approximation Method to find the underlying Gaussian Stochastic
        Processes which upon translation would yield the target non-Gaussian Stochastic Processes.

        :param distributions: An instance of the :py:mod:`UQpy` :class:`.Distributions` class defining the marginal
         distribution of the non-Gaussian stochastic process.
        :param time_interval: The value of time discretization.
        :param frequency_interval: The value of frequency discretization.
        :param n_time_intervals: The number of time discretizations.
        :param n_frequency_intervals: The number of frequency discretizations.
        :param target_correlation_function_non_gaussian: The target non-Gaussian autocorrelation function :math:`R_{NG}^T(\\tau)`
         stochastic processes.
         Either ``target_correlation_function_non_gaussian`` or ``target_power_spectrum_non_gaussian`` must be defined.
        :param target_power_spectrum_non_gaussian: The target non-Gaussian power spectrum :math:`S_{NG}^T(\\omega)`.
         Either ``target_correlation_function_non_gaussian`` or ``target_power_spectrum_non_gaussian`` must be defined.
        :param samples_non_gaussian: Samples of non-Gaussian stochastic processes.
         ``samples_non_gaussian`` is optional. If no samples are passed, the :class:`.InverseTranslation` class will
         compute the underlying Gaussian correlation using the ITAM.
        :param percentage_error: Percentage error that defines stopping criteria for ITAM. Default: 5.0
        """
        self.distributions = distributions
        self.time_interval = time_interval
        self.frequency_interval = frequency_interval
        self.n_time_intervals = n_time_intervals
        self.n_frequency_intervals = n_frequency_intervals
        self.samples_non_gaussian = samples_non_gaussian
        self.percentage_error = percentage_error
        if (  # only target_power_spectrum_non_gaussian is input
            target_correlation_function_non_gaussian is None
            and target_power_spectrum_non_gaussian is not None
        ):
            self.target_power_spectrum_non_gaussian = target_power_spectrum_non_gaussian
            self.target_correlation_function_non_gaussian = wiener_khinchin_transform(
                target_power_spectrum_non_gaussian, self.frequency, self.time
            )
        elif (  # only target_correlation_function_non_gaussian is input
            target_correlation_function_non_gaussian is not None
            and target_power_spectrum_non_gaussian is None
        ):
            self.target_correlation_function_non_gaussian = (
                target_correlation_function_non_gaussian
            )
            self.target_power_spectrum_non_gaussian = inverse_wiener_khinchin_transform(
                target_correlation_function_non_gaussian, self.frequency, self.time
            )
        else:
            raise RuntimeError(
                "UQpy: Exactly one of `target_correlation_function_non_gaussian` "
                "or `target_power_spectrum_non_gaussian` must be provided."
            )

        power_spectrum_gaussian, power_spectrum_non_gaussian = self._itam_power_spectrum()
        self.power_spectrum_gaussian: np.ndarray = power_spectrum_gaussian
        """The Gaussian power spectrum :math:`S_G(\\omega)` of the inverse translated Gaussian stochastic processes"""
        self.power_spectrum_non_gaussian: np.ndarray = power_spectrum_non_gaussian
        """The non-Gaussian power spectrum :math:`S_{NG}(\\omega)`.
         This is a reconstruction from the forward translation of ``power_spectrum_gaussian``"""
        self.auto_correlation_function_gaussian: np.ndarry = wiener_khinchin_transform(
            self.power_spectrum_gaussian, self.frequency, self.time
        )
        """The correlation function :math:`R_G(\\tau)` computed from ``power_spectrum_gaussian``."""
        self.correlation_function_gaussian: np.ndarry = (
            self.auto_correlation_function_gaussian
            / self.auto_correlation_function_gaussian[0]
        )
        """The correlation function of the inverse translated Gaussian stochastic processes."""
        self.samples_gaussian: np.ndarray = None
        """The inverse translated Gaussian samples from the non-Gaussian samples.
         The array is the same shape as ``samples_non_gaussian``."""

        self.frequency = np.arange(0, n_frequency_intervals) * frequency_interval
        self.time = np.arange(0, n_time_intervals) * time_interval
        self.logger = logging.getLogger(__name__)

        if self.samples_non_gaussian:
            self.run(self.samples_non_gaussian)

    def run(self, samples_non_gaussian):
        """Run the inverse translation approximation method to convert non-Gaussian smples to Gaussian samples

        :param samples_non_gaussian: Samples of non-Gaussian stochastic processes. If samples are provided at the
         initialization, then the run method is executed automatically. If no samples are passed at the
         initialization, the :class:`.InverseTranslation` class will compute the underlying Gaussian correlation using
         the ITAM and the run method needs to be manually executed by the user.
        """
        self.samples_non_gaussian = samples_non_gaussian.flatten()[:, np.newaxis]
        self.samples_gaussian = self._inverse_translate_non_gaussian_samples().reshape(
            samples_non_gaussian.shape
        )

    def _inverse_translate_non_gaussian_samples(self):
        samples_cdf = self.distributions.cdf(self.samples_non_gaussian)
        return Normal(loc=0.0, scale=1.0).icdf(samples_cdf)

    def _itam_power_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the Gaussian and reconstructed non-Gaussian power spectrum from the target non-Gaussian power spectrum

        :return: Gaussian power spectrum, reconstructed non-Gaussian power spectrum
        """
        target_S = self.target_power_spectrum_non_gaussian
        target_R = self.target_correlation_function_non_gaussian
        R_g_iterate = target_R
        S_g_iterate = target_S
        R_ng_iterate = np.zeros_like(R_g_iterate)
        r_ng_iterate = np.zeros_like(R_g_iterate)
        S_ng_iterate = np.zeros_like(S_g_iterate)
        non_gaussian_moments = getattr(self.distributions, "moments")()
        self.logger.info(
            "UQpy: Stochastic Process: Beginning Inverse Translation Approximation Method"
        )
        max_iter = 500
        i = 0
        error = np.inf
        while i < max_iter and error > self.percentage_error:
            R_g_iterate = wiener_khinchin_transform(
                S_g_iterate, self.frequency, self.time
            )
            for j in range(len(target_R)):
                r_ng_iterate[j] = correlation_distortion(
                    dist_object=self.distributions, rho=R_g_iterate[j] / R_g_iterate[0]
                )
            R_ng_iterate = (
                r_ng_iterate * non_gaussian_moments[1] + non_gaussian_moments[0] ** 2
            )
            S_ng_iterate = inverse_wiener_khinchin_transform(
                R_ng_iterate, self.frequency, self.time
            )

            error_1 = np.sum((target_S - S_ng_iterate) ** 2)
            error_2 = np.sum(target_S**2)
            error = 100 * np.sqrt(error_1 / error_2)

            ratio = target_S / S_ng_iterate
            ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
            S_g_next_iterate = (ratio**1.3) * S_g_iterate
            # Set negative entries to zero to eliminate numerical error of upgrading scheme
            S_g_next_iterate[S_g_next_iterate < 0] = 0
            S_g_iterate = S_g_next_iterate

            i += 1
        if error <= self.percentage_error:
            self.logger.info(
                "UQpy: Stochastic Process: Ended Inverse Translation Approximation Method due to error convergence"
            )
        if i == max_iter:
            self.logger.warning(
                "UQpy: Stochastic Process: Ended Inverse Translation Approximation Method after max iterations reached"
            )
        return S_g_iterate, S_ng_iterate

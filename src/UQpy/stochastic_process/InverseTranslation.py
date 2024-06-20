import numpy as np
from scipy import integrate
from UQpy.distributions import Distribution, Normal, MultivariateNormal
from UQpy.utilities import *
from UQpy.utilities.ValidationTypes import PositiveFloat
from UQpy.stochastic_process.supportive import (
    inverse_wiener_khinchin_transform,
    wiener_khinchin_transform,
)
from beartype import beartype
from line_profiler_pycharm import profile


# @beartype
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
        r"""A class to perform Iterative Translation Approximation Method to find the underlying Gaussian Stochastic
        Processes which upon translation would yield the target non-Gaussian Stochastic Processes.

        :param distributions: An instance of the :py:mod:`UQpy` :class:`.Distributions` class defining the marginal
         distribution of the non-Gaussian stochastic process.
        :param time_interval: The value of time discretization.
        :param frequency_interval: The value of frequency discretization.
        :param n_time_intervals: The number of time discretizations.
        :param n_frequency_intervals: The number of frequency discretizations.
        :param target_correlation_function_non_gaussian: The target non-Gaussian autocorrelation function :math:`R_{NG}^T(\tau)`
         Either ``target_correlation_function_non_gaussian`` or ``target_power_spectrum_non_gaussian`` must be defined.
        :param target_power_spectrum_non_gaussian: The target non-Gaussian power spectrum :math:`S_{NG}^T(\omega)`.
         Either ``target_correlation_function_non_gaussian`` or ``target_power_spectrum_non_gaussian`` must be defined.
        :param samples_non_gaussian: Samples of non-Gaussian stochastic processes.
         ``samples_non_gaussian`` is optional. If no samples are passed, the :class:`.InverseTranslation` class will
         compute the underlying Gaussian correlation using the ITAM.
        :param percentage_error: Percentage error that defines stopping criteria for ITAM. Default: 5.0

        :raise RuntimeError: If :math:`\Delta t > \pi / (\Delta \omega * n_{\omega})`, raise RuntimeError because aliasing will occur
        """
        if (
            target_power_spectrum_non_gaussian is None
            and target_correlation_function_non_gaussian is None
        ) or (
            target_power_spectrum_non_gaussian is not None
            and target_correlation_function_non_gaussian is not None
        ):
            raise RuntimeError(
                "UQpy: Exactly one of `target_correlation_function_non_gaussian` "
                "or `target_power_spectrum_non_gaussian` must be provided."
            )

        self.distributions = distributions
        self.time_interval = time_interval
        self.frequency_interval = frequency_interval
        self.n_time_intervals = n_time_intervals
        self.n_frequency_intervals = n_frequency_intervals

        cutoff_frequency = self.frequency_interval * self.n_frequency_intervals
        if (
            self.time_interval > np.pi / cutoff_frequency
        ):  # Equation 45 from Shinozuka 1991
            raise RuntimeError(
                "UQpy: `time_interval` is too large for cutoff frequency. Aliasing will occur."
            )

        self.frequency = np.arange(0, n_frequency_intervals) * frequency_interval
        self.time = np.arange(0, n_time_intervals) * time_interval
        self.logger = logging.getLogger(__name__)

        self.target_power_spectrum_non_gaussian = (
            target_power_spectrum_non_gaussian
            if (target_power_spectrum_non_gaussian is not None)
            else inverse_wiener_khinchin_transform(
                target_correlation_function_non_gaussian, self.frequency, self.time
            )
        )
        self.target_correlation_function_non_gaussian = (
            target_correlation_function_non_gaussian
            if (target_correlation_function_non_gaussian is not None)
            else wiener_khinchin_transform(
                target_power_spectrum_non_gaussian, self.frequency, self.time
            )
        )
        self.samples_non_gaussian = samples_non_gaussian
        self.percentage_error = percentage_error

        s_g, s_ng = self._itam_power_spectrum()
        self.power_spectrum_gaussian: np.ndarray = s_g
        r"""The Gaussian power spectrum :math:`S_G(\omega)` of the inverse translated Gaussian stochastic processes."""
        self.power_spectrum_non_gaussian: np.ndarray = s_ng
        r"""The non-Gaussian power spectrum :math:`S_{NG}(\omega)`.
         This is a reconstruction from the forward translation of ``power_spectrum_gaussian``."""
        self.auto_correlation_function_gaussian: np.ndarray = wiener_khinchin_transform(
            self.power_spectrum_gaussian, self.frequency, self.time
        )
        r"""The correlation function :math:`R_G(\tau)` computed from ``power_spectrum_gaussian``."""
        self.correlation_function_gaussian: np.ndarray = (
            self.auto_correlation_function_gaussian
            / self.auto_correlation_function_gaussian[0]
        )
        """The correlation function of the inverse translated Gaussian stochastic processes."""
        self.samples_gaussian: np.ndarray = None
        """The inverse translated Gaussian samples from the non-Gaussian samples.
         This array is the same shape as ``samples_non_gaussian``."""

        if self.samples_non_gaussian:
            self.run(self.samples_non_gaussian)

    def run(self, samples_non_gaussian: np.ndarray):
        """Run the inverse translation approximation method to convert non-Gaussian samples to Gaussian samples

        :param samples_non_gaussian: Samples of non-Gaussian stochastic processes.
         If samples are provided at the initialization, then the run method is executed automatically.
         If no samples are passed at the initialization, the :class:`.InverseTranslation` class will compute the
         underlying Gaussian correlation using the ITAM and the run method needs to be manually executed by the user.
        """
        self.samples_non_gaussian = samples_non_gaussian.flatten()[:, np.newaxis]
        self.samples_gaussian = self._inverse_translate_non_gaussian_samples().reshape(
            samples_non_gaussian.shape
        )

    def _inverse_translate_non_gaussian_samples(self):
        samples_cdf = self.distributions.cdf(self.samples_non_gaussian)
        return Normal(loc=0.0, scale=1.0).icdf(samples_cdf)

    @profile
    def _itam_power_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the Gaussian and reconstructed non-Gaussian power spectrum from the target non-Gaussian power spectrum

        :return: Gaussian power spectrum, reconstructed non-Gaussian power spectrum
        """
        target_S = self.target_power_spectrum_non_gaussian
        target_R = self.target_correlation_function_non_gaussian
        R_g_iterate = target_R
        S_g_iterate = target_S
        R_ng_iterate = np.zeros_like(R_g_iterate)
        S_ng_iterate = np.zeros_like(S_g_iterate)
        mean_ng, variance_ng = self.distributions.moments(moments2return="mv")
        self.logger.info(
            "UQpy: Stochastic Process: Beginning Inverse Translation Approximation Method"
        )
        max_iter = 500
        i = 0
        error = np.inf
        normal = Normal(scale=float(variance_ng))

        @profile
        def integrand(x1, x2, rho):
            cov = np.array([[var, rho], [rho, var]])
            bivariate_normal = MultivariateNormal(np.array([0.0, 0.0]), cov)
            return (
                self.distributions.icdf(normal.cdf(x1))
                * self.distributions.icdf(normal.cdf(x2))
                * bivariate_normal.pdf(np.array([x1, x2]))
            )

        bounds = 4
        self.logger.info(
            f"UQpy: Stochastic Process: Iteration={i} / {max_iter} Error={error} ErrorThreshold={self.percentage_error}"
        )
        while i < max_iter and error > self.percentage_error:
            R_g_iterate = wiener_khinchin_transform(
                S_g_iterate, self.frequency, self.time
            )
            var = R_g_iterate[0]
            for j in range(len(target_R)):
                R_ng_iterate[j], _ = integrate.dblquad(
                    lambda x1, x2: integrand(x1, x2, rho=R_g_iterate[j] / var),
                    -bounds,
                    bounds,
                    -bounds,
                    bounds,
                )

            R_ng_iterate = (R_ng_iterate * variance_ng) + (mean_ng**2)
            S_ng_iterate = inverse_wiener_khinchin_transform(
                R_ng_iterate, self.frequency, self.time
            )

            error = 100 * np.sqrt(
                np.sum((target_S - S_ng_iterate) ** 2) / np.sum(target_S**2)
            )

            ratio = target_S / S_ng_iterate
            ratio = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
            ratio = np.clip(
                ratio, 0, None
            )  # Set negative entries to zero to avoid complex numbers after fractional exponent
            S_g_next_iterate = (ratio**1.3) * S_g_iterate
            S_g_iterate = np.clip(S_g_next_iterate, 0, None)

            i += 1
            self.logger.info(
                f"UQpy: Stochastic Process: Iteration={i} / {max_iter} Error={error} ErrorThreshold={self.percentage_error}"
            )

        self.logger.info(
            f"UQpy: Stochastic Process: Ended Inverse Translation Approximation Method"
        )
        if error > self.percentage_error:
            self.logger.warning(
                "UQpy: Stochastic Process: InverseTranslation may have undesirably large error"
            )
        return S_g_iterate, S_ng_iterate

    # def _bivariate_normal_pdf(
    #     self, x1: float, x2: float, rho: float, sigma_squared: float = 1.0
    # ) -> np.ndarray:
    #     """The probability density function of a zero-mean bivariate normal distribution with correlation ``rho``
    #     As defined by Equation 17 of Shields 2011
    #
    #     :param x1: First coordinate
    #     :param x2: Second coordinate
    #     :param rho: Normalized correlation function
    #     :return: :math:`\phi(x_1, x_2; \rho)`
    #     """
    #     rho = np.clip(rho, -0.999, 0.999)
    #     coefficient = 1 / (2 * np.pi * sigma_squared * np.sqrt(1 - rho**2))
    #     numerator = x1**2 + x2**2 - (2 * rho * x1 * x2)
    #     denominator = 2 * sigma_squared * (1 - rho)
    #     return coefficient * np.exp(-numerator / denominator)
